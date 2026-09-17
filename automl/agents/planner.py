"""Profile bakarak preprocessing planini uretir."""
import math
import time
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from automl import llm
from automl.agents.base import Agent
from automl.agents.modeler import _candidate_models, _cv_bolucu
# Nadir kategori esigi encoder'in parametresi, preprocessor'da tanimli.
from automl.agents.preprocessor import NADIR_KATEGORI_ESIGI, pipeline_kur
# ID orani profiler'da test ediliyor, esik de orada tanimli: gerekce
# metninde ayni sabit kullanilsin ki basilan esik koddan sapmasin.
from automl.agents.profiler import (GIZLI_SAYISAL_ESIGI, ID_ORAN_ESIGI,
                                    ana_metrik)
from automl.memory.store import benzer_runlar
from automl.schemas import (ColumnDecision, ColumnProfile, DataProfile,
                            KararTipi, PreprocessingPlan, RunState)

SYSTEM_PROMPT = (
    "Sen bir ML preprocessing uzmanisin. Verilen veri profiline bakarak "
    "preprocessing kararlari oner."
)

# Karar esikleri: hem karsilastirmada hem gerekce metninde ayni sabit
# kullanilir, boylece basilan esik koddan sapamaz.
NULL_ESIGI = 0.5          # bu oranin ustunde null olan kolon atilir
KARDINALITE_ESIGI = 50    # bundan fazla kategori: one-hot patlar, strateji
# Train satirlarinin bu oranindan fazlasi train'de yalnizca BIR kez gorulen
# degerlere sahipse kolon atilir. Gerekce: toplama bu satirlarin hepsini tek
# nadir kategorisine, frequency encoding hepsini ayni degere (1/n) indirir;
# iki strateji de neredeyse sabit bir kolon uretir ve test'teki degerler
# buyuk ihtimalle train'de hic gorulmemistir. %90, "neredeyse her satir
# farkli" durumunu yakalar; daha dusuk bir esik, tekrar eden kategorilerin
# tasidigi bilgiyi de cope atardi.
TEKIL_SATIR_ESIGI = 0.90
# Yuksek kardinaliteli kolonun katkisi train icinde CV ile olculur. Kolon,
# ancak kolonlu CV ortalamasi kolonsuzu
#     KATKI_KATSAYISI * sqrt((std_kolonlu^2 + std_kolonsuz^2) / 2)
# kadar gecerse tutulur. Referans, iki surumun fold'lar arasi tipik
# oynakligidir (havuzlanmis std). Ortalamanin standart hatasi
# (std / sqrt(k)) kullanilmadi: fold'lar ayni satirlari paylastigi icin
# skorlari bagimsiz degil, std / sqrt(k) belirsizligi oldugundan kucuk
# gosterir. 1.0 katsayi 5 fold'da yaklasik 2.2 standart hataya denk gelir:
# yaygin "2 standart hata" kuralina yakin, fold bagimliligi yuzunden de
# temkinli. Telco'da oynaklik +-0.011 iken 0.0001'lik fark bu esigin cok
# altinda kalirdi; gurultu kolon tutma karari veremez.
KATKI_KATSAYISI = 1.0
PCA_KOLON_ESIGI = 10      # bundan fazla sayisal kolon varsa PCA aday
PCA_SATIR_ESIGI = 50      # PCA icin gereken en az satir sayisi

# Imputation / scaling / encoding secimlerinin gerekceleri.
ADIM_GEREKCELERI = {
    "numeric_imputation":
        "median: aykırı değerlere dayanıklı, ortalamayı kaydırmaz",
    "categorical_imputation":
        "most_frequent: kategorikte ortalama yok, en sık değer en az "
        "bilgi bozar",
    "scaling":
        "standard: kolonlar farklı ölçeklerde, mesafe ve gradyan tabanlı "
        "modeller ölçeğe duyarlı",
    "encoding":
        "onehot: kategoriler ordinal değil, sayı vermek modele yanlış "
        "sıralama öğretir",
}


def _pca_gerekcesi(n_sayisal: int, n_satir: int, use_pca: bool,
                   n_components: int | None) -> str:
    "PCA'nin neden acildigini/kapandigini tetikleyen degerlerle anlatir."
    if use_pca:
        return (f"açıldı ({n_components} bileşene indirildi) — "
                f"{n_sayisal} sayısal kolon > eşik {PCA_KOLON_ESIGI}, "
                f"{n_satir} satır > eşik {PCA_SATIR_ESIGI}")

    # Kapaliysa hangi kosul(lar) tutmadi, tek tek yaz.
    sebepler = []
    if n_sayisal <= PCA_KOLON_ESIGI:
        sebepler.append(f"{n_sayisal} sayısal kolon, eşik "
                        f"{PCA_KOLON_ESIGI}'un üzerinde değil")
    if n_satir <= PCA_SATIR_ESIGI:
        sebepler.append(f"{n_satir} satır, eşik "
                        f"{PCA_SATIR_ESIGI}'nin üzerinde değil")
    return "kapalı — " + "; ".join(sebepler)


def _donusum_gerekcesi(c: ColumnProfile) -> tuple[str, str]:
    "Metin tipinde gelip sayiya cevrilen kolonun (sebep, tetik) metni."
    oran = c.donusum_orani or 0.0
    sebep = (f"metin tipinde (dtype={c.ham_dtype}) geldi ama değerleri "
             f"sayı: gizli sayısal kolon, sayıya çevrilip sayısal "
             f"pipeline'a gider")
    tetik = (f"sayıya çevrilebilen dolu değer oranı %{oran * 100:.1f} >= "
             f"GIZLI_SAYISAL_ESIGI %{GIZLI_SAYISAL_ESIGI * 100:g}")
    if c.n_donusmeyen:
        ornekler = ", ".join(c.donusmeyen_ornekler)
        tetik += (f"; {c.n_donusmeyen} değer ({ornekler}) eksik sayılıp "
                  f"imputation'a bırakıldı")
    return sebep, tetik


def _yuksek_kardinalite_karari(c: ColumnProfile,
                               X_train) -> tuple[KararTipi, str, str]:
    """KARDINALITE_ESIGI'ni asan kategorik kolon icin strateji secer.

    Sira: neredeyse her satir farkli -> at; nadir toplama sonrasi esigin
    altina iniyor -> topla + one-hot; inmiyor -> frequency encoding.
    Sayimlar TRAIN setinden yapilir: encoder'lar da sadece train'de fit
    edilir, boylece basilan sayilar pipeline'in ogrendigiyle ayni olur.
    Eksik degerler sayilmaz; imputation onlari en sik kategoriye atar.
    """
    if X_train is None or c.name not in X_train:
        raise RuntimeError("plan: yuksek kardinalite karari train setinden "
                           "verilir, once split calismali")
    seri = X_train[c.name]
    n = len(seri)
    sayim = seri.value_counts()
    n_kat = len(sayim)
    nadir = int((sayim < NADIR_KATEGORI_ESIGI * n).sum())
    kalan = n_kat - nadir + (1 if nadir else 0)
    tekil_oran = float(sayim[sayim == 1].sum()) / max(n, 1)

    nadir_esik = f"%{NADIR_KATEGORI_ESIGI * 100:g} (NADIR_KATEGORI_ESIGI)"
    kard = (f"tüm veride {c.n_unique} kategori > eşik {KARDINALITE_ESIGI} "
            f"(KARDINALITE_ESIGI)")

    if tekil_oran > TEKIL_SATIR_ESIGI:
        return (
            "drop",
            "neredeyse her satır farklı: toplama bu satırları tek nadir "
            "kategoride, frequency encoding aynı değerde birleştirir; kolon "
            "bilgi taşımaz",
            f"train'de {n_kat} kategori / {n} satır; tek kez görülen değerli "
            f"satır oranı %{tekil_oran * 100:.0f} > TEKIL_SATIR_ESIGI "
            f"%{TEKIL_SATIR_ESIGI * 100:g}",
        )
    if kalan <= KARDINALITE_ESIGI:
        return (
            "nadir_toplama",
            f"{kard}, ama train'deki {n_kat} kategoriden {nadir} tanesi "
            f"{nadir_esik} altında",
            f"toplama sonrası {kalan} kategori kaldı (<= eşik "
            f"{KARDINALITE_ESIGI}), one-hot makul",
        )
    return (
        "frekans",
        f"{kard}; train'deki {n_kat} kategoriden sadece {nadir} tanesi "
        f"{nadir_esik} altında, toplama yetmiyor",
        f"toplama sonrası {kalan} kategori > eşik {KARDINALITE_ESIGI}; "
        f"frequency encoding kolon başına tek sayısal kolon üretir",
    )


@dataclass
class _KatkiOlcumu:
    "Bir kolonun train icinde CV ile olculen katkisi."
    metrik: str
    model_kolonlu: str
    model_kolonsuz: str
    kolonlu: tuple[float, float]      # (cv ortalama, cv std)
    kolonsuz: tuple[float, float]
    sure_sn: float

    @property
    def fark(self) -> float:
        return self.kolonlu[0] - self.kolonsuz[0]

    @property
    def referans(self) -> float:
        "Iki surumun havuzlanmis fold oynakligi."
        return math.sqrt((self.kolonlu[1] ** 2 + self.kolonsuz[1] ** 2) / 2)

    @property
    def esik(self) -> float:
        return KATKI_KATSAYISI * self.referans

    @property
    def anlamli(self) -> bool:
        return self.fark > self.esik

    def metin(self) -> str:
        model = self.model_kolonlu
        if self.model_kolonsuz != self.model_kolonlu:
            model += f", kolonsuz sürümde özellik yok: {self.model_kolonsuz}"
        return (
            f"CV ({model}, {self.metrik}): kolonlu {self.kolonlu[0]:.3f} vs "
            f"kolonsuz {self.kolonsuz[0]:.3f}, fark {self.fark:+.3f} "
            f"{'>' if self.anlamli else '<='} oynaklık eşiği {self.esik:.3f} "
            f"(KATKI_KATSAYISI {KATKI_KATSAYISI:g} × ±{self.referans:.3f})"
        )


def _cv_skoru(pl: PreprocessingPlan, X, y,
              p: DataProfile) -> tuple[str, float, float]:
    """Plandaki ozelliklerle referans modelin CV skoru: (model, ort, std).

    On isleme model ile ayni Pipeline'da: her fold'da SADECE o fold'un
    train kisminda fit edilir. Ozellik yoksa hicbir sey ogrenmeyen
    Baseline olculur.
    """
    ozellikler = (pl.numeric_cols + pl.categorical_cols
                  + pl.nadir_toplama_cols + pl.frekans_cols)
    havuz = _candidate_models(p.task_type, p.n_rows)
    if ozellikler:
        ad = ("LinearRegression" if p.task_type == "regression"
              else "LogisticRegression")
        model = Pipeline([("hazirlik", pipeline_kur(pl)),
                          ("model", havuz[ad])])
    else:
        ad = "Baseline"
        model = havuz[ad]
    skorlar = cross_val_score(model, X, y, cv=_cv_bolucu(p.task_type, y),
                              scoring=ana_metrik(p))
    return ad, float(np.mean(skorlar)), float(np.std(skorlar))


def _katki_olc(kolon: str, strateji: KararTipi, taban: PreprocessingPlan,
               X_train, y_train, p: DataProfile) -> _KatkiOlcumu:
    """Kolonu dahil eden ve etmeyen iki surumu train icinde CV ile olcer.

    SADECE X_train / y_train kullanilir; test seti hic gorulmez. Kolon,
    tutulursa alacagi stratejiyle kodlanir. Varsayilan on isleme (median,
    most_frequent, PCA yok) kullanilir ki olcum iterasyon stratejisinden
    bagimsiz ve her iterasyonda ayni olsun.
    """
    baslangic = time.perf_counter()
    alan = {"nadir_toplama": "nadir_toplama_cols",
            "frekans": "frekans_cols"}[strateji]
    kolonlu = taban.model_copy(update={alan: [kolon]})
    ad_ile, ort_ile, std_ile = _cv_skoru(kolonlu, X_train, y_train, p)
    ad_siz, ort_siz, std_siz = _cv_skoru(taban, X_train, y_train, p)
    return _KatkiOlcumu(
        metrik=ana_metrik(p), model_kolonlu=ad_ile, model_kolonsuz=ad_siz,
        kolonlu=(ort_ile, std_ile), kolonsuz=(ort_siz, std_siz),
        sure_sn=time.perf_counter() - baslangic,
    )


STRATEJI_ADLARI = {"nadir_toplama": "nadir toplama + onehot",
                   "frekans": "frequency encoding"}


def _katkiya_gore_kesinlestir(gecici: ColumnDecision, c: ColumnProfile,
                              taban: PreprocessingPlan, state: RunState,
                              p: DataProfile) -> ColumnDecision:
    "Strateji secilmis kolonu, CV'deki katkisina gore tutar ya da atar."
    if state.y_train is None:
        return gecici     # hedef yok (clustering): katki olculemez
    olcum = _katki_olc(c.name, gecici.decision, taban, state.X_train,
                       state.y_train, p)
    print(f"    katki olcumu: {c.name} -> "
          f"{'tutuldu' if olcum.anlamli else 'atildi'} "
          f"(fark {olcum.fark:+.3f}, esik {olcum.esik:.3f}, "
          f"{olcum.sure_sn:.2f} sn)")

    if olcum.anlamli:
        return gecici.model_copy(update={
            "reason": f"{gecici.reason}; katkısı train içinde CV ile "
                      f"ölçüldü, sinyal taşıyor",
            "trigger": f"{olcum.metin()}; {gecici.trigger}",
        })
    return gecici.model_copy(update={
        "decision": "drop",
        "reason": f"{c.n_unique} kategori > eşik {KARDINALITE_ESIGI}; "
                  f"katkısı train içinde CV ile ölçüldü "
                  f"({STRATEJI_ADLARI[gecici.decision]} ile kodlanarak)",
        "trigger": f"{olcum.metin()} — sinyal taşımıyor",
    })


def plan(state: RunState) -> RunState:
    "Profile bakarak preprocessing kararlarini otomatik uretir."
    p = state.profile
    if p is None:
        raise RuntimeError("plan: once profile adimi calismali")

    notes = []

    numeric_cols = []
    categorical_cols = []
    nadir_toplama_cols = []
    frekans_cols = []
    drop_cols = []
    kararlar: list[ColumnDecision] = []
    # Strateji secilmis ama katkisi henuz olculmemis yuksek kardinaliteli
    # kolonlar: (kararlar icindeki sira, profil). Taban ozellikler dongu
    # bitince belli olur, olcum oradan sonra yapilir.
    bekleyenler: list[tuple[int, ColumnProfile]] = []

    # Esik metinleri tek yerde uretilir, her gerekcede ayni sekilde gecer.
    null_esik_metni = f"eşik %{NULL_ESIGI*100:.0f}"
    kard_esik_metni = f"eşik {KARDINALITE_ESIGI}"

    for c in p.columns:
        if c.name == p.target:
            kararlar.append(ColumnDecision(
                name=c.name, inferred_type=c.inferred_type,
                decision="target",
                reason="hedef kolon, özellik değil — tahmin edilen değer, "
                       "modele girdi olarak verilmez",
                trigger=f"görev tipi {p.task_type}",
            ))
            continue

        if c.null_ratio > NULL_ESIGI:
            drop_cols.append(c.name)
            notes.append(f"{c.name}: %{c.null_ratio*100:.0f} bos, atildi")
            kararlar.append(ColumnDecision(
                name=c.name, inferred_type=c.inferred_type, decision="drop",
                reason="eksik değer oranı çok yüksek, güvenilir doldurma "
                       "yapılamaz",
                trigger=f"null oranı %{c.null_ratio*100:.0f} > "
                        f"{null_esik_metni}",
            ))
            continue

        if c.inferred_type in ("text", "datetime"):
            drop_cols.append(c.name)
            notes.append(f"{c.name}: {c.inferred_type} tipi, atildi")
            if c.inferred_type == "text":
                sebep = ("serbest metin, özellik çıkarımı olmadan modele "
                         "giremez")
                tetik = (f"tip=text, {c.n_unique} eşsiz değer / "
                         f"{p.n_rows} satır")
            else:
                sebep = ("tarih kolonu, özellik çıkarımı (yıl/ay/gün) "
                         "olmadan modele giremez")
                tetik = f"tip=datetime, dtype={c.dtype}"
            kararlar.append(ColumnDecision(
                name=c.name, inferred_type=c.inferred_type, decision="drop",
                reason=sebep, trigger=tetik,
            ))
            continue

        if (c.inferred_type == "categorical"
                and c.n_unique > KARDINALITE_ESIGI):
            # Uc durum CV'ye gitmeden atilir; digerleri icin strateji
            # secilir, tutulup tutulmayacagi katki olcumunde kesinlesir.
            karar, sebep, tetik = _yuksek_kardinalite_karari(c,
                                                             state.X_train)
            if karar == "drop":
                drop_cols.append(c.name)
                notes.append(f"{c.name}: {c.n_unique} essiz kategori, "
                             f"neredeyse her satir farkli, atildi")
            else:
                bekleyenler.append((len(kararlar), c))
            kararlar.append(ColumnDecision(
                name=c.name, inferred_type=c.inferred_type, decision=karar,
                reason=sebep, trigger=tetik,
            ))
            continue

        if c.is_probable_id:
            drop_cols.append(c.name)
            notes.append(f"{c.name}: kimlik kolonu, atildi")
            oran = c.n_unique / max(p.n_rows, 1)
            tetik = (f"{c.n_unique} eşsiz / {p.n_rows} satır = "
                     f"%{oran*100:.0f} (eşik %{ID_ORAN_ESIGI*100:.0f})")
            if c.id_ardisik:
                tetik += ", değerler ardışık"
            kararlar.append(ColumnDecision(
                name=c.name, inferred_type=c.inferred_type, decision="drop",
                reason="satır başına benzersiz değer, kimlik kolonu — "
                       "bilgi taşımıyor",
                trigger=tetik,
            ))
            continue

        if c.inferred_type == "numeric":
            numeric_cols.append(c.name)
            if c.donusturuldu:
                sebep, tetik = _donusum_gerekcesi(c)
            else:
                sebep = "sayısal tip, sayısal pipeline'a gider"
                tetik = (f"null oranı %{c.null_ratio*100:.0f}, "
                         f"{null_esik_metni} altında")
            kararlar.append(ColumnDecision(
                name=c.name, inferred_type=c.inferred_type,
                decision="numeric", reason=sebep, trigger=tetik,
            ))
        else:
            categorical_cols.append(c.name)
            kararlar.append(ColumnDecision(
                name=c.name, inferred_type=c.inferred_type,
                decision="categorical",
                reason="kategorik tip, kategorik pipeline'a gider",
                trigger=f"{c.n_unique} eşsiz değer, kardinalite "
                        f"{kard_esik_metni} altında",
            ))

    # Yuksek kardinaliteli kolonlarin katkisi: her biri ayni tabana (dusuk
    # kardinaliteli ozellikler) karsi, birbirinden bagimsiz olculur.
    taban = PreprocessingPlan(numeric_cols=list(numeric_cols),
                              categorical_cols=list(categorical_cols))
    for sira, c in bekleyenler:
        karar = _katkiya_gore_kesinlestir(kararlar[sira], c, taban, state, p)
        kararlar[sira] = karar
        {"drop": drop_cols, "nadir_toplama": nadir_toplama_cols,
         "frekans": frekans_cols}[karar.decision].append(c.name)
        notes.append(f"{c.name}: {c.n_unique} essiz kategori, "
                     f"karar={karar.decision}")
    # Sonradan atilan kolonlar da veri setindeki sirada dursun.
    kolon_sirasi = {c.name: i for i, c in enumerate(p.columns)}
    drop_cols.sort(key=lambda ad: kolon_sirasi[ad])

    use_pca = (len(numeric_cols) > PCA_KOLON_ESIGI
               and p.n_rows > PCA_SATIR_ESIGI)
    n_components = min(10, len(numeric_cols)) if use_pca else None
    if use_pca:
        notes.append(f"{len(numeric_cols)} sayisal kolon icin PCA acildi")
    pca_reason = _pca_gerekcesi(len(numeric_cols), p.n_rows, use_pca,
                                n_components)

    numeric_imputation = "median"
    categorical_imputation = "most_frequent"
    step_reasons = dict(ADIM_GEREKCELERI)

    # Strateji katmani: "varsayilan" disinda kurallari oynatir.
    # Onceki iterasyon eşigi gecemediyse orchestrator strateji degistirir.
    if state.strateji == "pca_ters":
        if use_pca:
            use_pca = False
            n_components = None
            notes.append("strateji: PCA kapatildi (tersine cevrildi)")
            pca_reason = (
                f"strateji 'pca_ters': kural PCA'yı açmıştı "
                f"({len(numeric_cols)} sayısal kolon), önceki iterasyon "
                f"eşiği geçemediği için tersine çevrilip kapatıldı"
            )
        elif len(numeric_cols) >= 2:
            use_pca = True
            n_components = min(10, len(numeric_cols))
            notes.append(
                f"strateji: PCA acildi (tersine cevrildi, "
                f"n_components={n_components})"
            )
            pca_reason = (
                f"strateji 'pca_ters': kural PCA'yı kapatmıştı, önceki "
                f"iterasyon eşiği geçemediği için tersine çevrilip açıldı; "
                f"{n_components} bileşene indirildi"
            )
        else:
            notes.append("strateji: PCA acilamadi, yeterli sayisal kolon yok")
            pca_reason = (
                f"strateji 'pca_ters': PCA açılmak istendi ama "
                f"{len(numeric_cols)} sayısal kolon ile açılamaz "
                f"(en az 2 gerekir)"
            )
    elif state.strateji == "imputation_degis":
        numeric_imputation = "mean"
        categorical_imputation = "constant"
        notes.append("strateji: imputation median->mean, "
                     "most_frequent->constant")
        step_reasons["numeric_imputation"] = (
            "mean: strateji 'imputation_degis' önceki iterasyon eşiği "
            "geçemediği için medyan yerine ortalamayı denedi"
        )
        step_reasons["categorical_imputation"] = (
            "constant: strateji 'imputation_degis' en sık değer yerine "
            "sabit değer atamayı denedi"
        )

    state.plan = PreprocessingPlan(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        nadir_toplama_cols=nadir_toplama_cols,
        frekans_cols=frekans_cols,
        drop_cols=drop_cols,
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        scaling="standard",
        encoding="onehot",
        use_pca=use_pca,
        n_components=n_components,
        notes=notes,
        column_decisions=kararlar,
        pca_reason=pca_reason,
        step_reasons=step_reasons,
    )
    return state


def _profil_ozeti(p: DataProfile, kural_plan: PreprocessingPlan) -> str:
    "LLM'e verilecek profil + kural-tabanli plan ozeti."
    satirlar = [
        f"Satir sayisi: {p.n_rows}",
        f"Kolon sayisi: {p.n_cols}",
        f"Hedef kolon: {p.target}",
        f"Gorev tipi: {p.task_type}",
        "",
        "Kolonlar (ad | tip | essiz | null orani):",
    ]
    for c in p.columns:
        satirlar.append(
            f"  {c.name} | {c.inferred_type} | {c.n_unique} | "
            f"{c.null_ratio:.2f}"
        )

    if p.high_correlations:
        satirlar.append("")
        satirlar.append("Yuksek korelasyonlar (|r| >= 0.85):")
        for a, b, r in p.high_correlations:
            satirlar.append(f"  {a} <-> {b} r={r:+.3f}")

    satirlar += [
        "",
        "Kural-tabanli planin onerisi:",
        f"  numeric_cols: {kural_plan.numeric_cols}",
        f"  categorical_cols: {kural_plan.categorical_cols}",
        f"  nadir_toplama_cols: {kural_plan.nadir_toplama_cols}",
        f"  frekans_cols: {kural_plan.frekans_cols}",
        f"  drop_cols: {kural_plan.drop_cols}",
        f"  numeric_imputation: {kural_plan.numeric_imputation}",
        f"  categorical_imputation: {kural_plan.categorical_imputation}",
        f"  scaling: {kural_plan.scaling}",
        f"  encoding: {kural_plan.encoding}",
        f"  use_pca: {kural_plan.use_pca} (n_components="
        f"{kural_plan.n_components})",
        "",
        "Bu profile bakarak preprocessing planini JSON olarak dondur. "
        "Sadece JSON dondur, baska aciklama yazma. Alanlar: numeric_cols, "
        "categorical_cols, nadir_toplama_cols (nadir kategorileri toplanip "
        "one-hot), frekans_cols (frequency encoding), drop_cols, "
        "numeric_imputation, "
        "categorical_imputation, scaling, encoding, use_pca, n_components, "
        "notes. Kolon adlarini AYNEN yukaridaki listeden kullan, yeni kolon "
        "adi uydurma. Hedef kolonu ozellik listelerine koyma.",
    ]
    return "\n".join(satirlar)


def _gecmis_ozeti(p: DataProfile) -> tuple[str, list[str]]:
    """Benzer gecmis run'lari LLM baglami + insan okur not olarak dondurur.

    Donen: (llm_prompt_parcasi, notes_satirlari). Gecmis yoksa ikisi de bos.
    """
    try:
        benzerler = benzer_runlar(p)
    except Exception as e:
        print(f"   ! gecmis run'lar okunamadi: {type(e).__name__}: {e}")
        return "", []

    if not benzerler:
        return "", []

    prompt_satirlari = ["", "Benzer veride daha once denenen planlar:"]
    not_satirlari = []
    for kayit in benzerler:
        r = kayit["run"].get("result") or {}
        pl = kayit["run"].get("plan") or {}
        if not r or not pl:
            continue
        ozet = (
            f"benzerlik={kayit['benzerlik']:.2f} | "
            f"PCA={pl.get('use_pca')}({pl.get('n_components')}) | "
            f"imputation={pl.get('numeric_imputation')}/"
            f"{pl.get('categorical_imputation')} | "
            f"model={r.get('model_name')} | "
            f"{r.get('metric_name')}={r.get('metric_value'):.4f}"
        )
        prompt_satirlari.append(f"  {ozet}")
        not_satirlari.append(f"gecmis: {ozet}")

    if not not_satirlari:
        return "", []

    prompt_satirlari.append(
        "  Bu gecmis sonuclari dikkate al, daha iyi skor veren plana yaklas."
    )
    return "\n".join(prompt_satirlari), not_satirlari


def _kolonlar_gecerli(oneri: PreprocessingPlan, p: DataProfile) -> bool:
    "LLM'in onerdigi kolon adlari gercekten veride var mi?"
    gecerli = {c.name for c in p.columns if c.name != p.target}
    onerilen = (oneri.numeric_cols + oneri.categorical_cols
                + oneri.nadir_toplama_cols + oneri.frekans_cols
                + oneri.drop_cols)
    for ad in onerilen:
        if ad not in gecerli:
            print(f"   ! LLM uydurma kolon adi verdi: {ad!r}, plan reddedildi")
            return False
    if not (oneri.numeric_cols or oneri.categorical_cols
            or oneri.nadir_toplama_cols or oneri.frekans_cols):
        print("   ! LLM bos ozellik listesi verdi, plan reddedildi")
        return False
    return True


def _gerekceleri_devret(oneri: PreprocessingPlan,
                        kural: PreprocessingPlan) -> None:
    """LLM plani kabul edilince karar gerekcelerini ona tasir.

    LLM gerekce uretmiyor (prompt'ta bu alanlar istenmiyor). Kural-tabanli
    planin gerekcesi korunur; LLM farkli karar verdiyse bu acikca yazilir,
    gerekce uydurulmaz.
    """
    llm_karari: dict[str, KararTipi] = {}
    for ad in oneri.numeric_cols:
        llm_karari[ad] = "numeric"
    for ad in oneri.categorical_cols:
        llm_karari[ad] = "categorical"
    for ad in oneri.nadir_toplama_cols:
        llm_karari[ad] = "nadir_toplama"
    for ad in oneri.frekans_cols:
        llm_karari[ad] = "frekans"
    for ad in oneri.drop_cols:
        llm_karari[ad] = "drop"

    yeni: list[ColumnDecision] = []
    for k in kural.column_decisions:
        if k.decision == "target":
            yeni.append(k)          # hedef kolon LLM'in kararina tabi degil
            continue
        # LLM kolonu hic anmadiysa ozellik listesine girmemis demektir.
        karar = llm_karari.get(k.name, "drop")
        if karar == k.decision:
            yeni.append(k)
            continue
        yeni.append(k.model_copy(update={
            "decision": karar,
            "reason": (f"LLM önerisi kural kararının ({k.decision}) yerine "
                       f"bunu seçti; kural gerekçesi: {k.reason}"),
            "trigger": k.trigger,
        }))

    oneri.column_decisions = yeni
    oneri.pca_reason = (
        f"LLM planı: use_pca={oneri.use_pca} "
        f"(n_components={oneri.n_components}); "
        f"kural gerekçesi -> {kural.pca_reason}"
    )
    oneri.step_reasons = dict(kural.step_reasons)


class PlannerAgent(Agent):
    """Preprocessing planini uretir."""

    name = "planner"

    def run(self, state: RunState) -> RunState:
        # 1) Kural-tabanli plan her zaman calisir, guvenli taban budur.
        state = plan(state)
        kural_plan = state.plan
        p = state.profile
        if kural_plan is None or p is None:
            return state

        # 2) Benzer gecmis run'lardan baglam topla.
        gecmis_prompt, gecmis_notlar = _gecmis_ozeti(p)

        # 3) LLM varsa gecmisi de vererek ikinci bir gorus al.
        oneri = None
        if llm.is_available():
            oneri = llm.ask_json(
                SYSTEM_PROMPT,
                _profil_ozeti(p, kural_plan) + gecmis_prompt,
                PreprocessingPlan,
            )
        elif gecmis_notlar:
            # LLM yok: gecmis bilgisi sadece bilgi amacli nota yazilir.
            kural_plan.notes.extend(gecmis_notlar)

        # 4) Gecerliyse LLM planini kullan, degilse kural-tabanliya dus.
        if oneri is not None and _kolonlar_gecerli(oneri, p):
            oneri.notes.append("LLM onerisi kullanildi")
            _gerekceleri_devret(oneri, kural_plan)
            state.plan = oneri
        else:
            kural_plan.notes.append("kural-tabanli plan kullanildi")
            state.plan = kural_plan

        return state
