"""Profile bakarak preprocessing planini uretir."""
from automl import llm
from automl.agents.base import Agent
# ID orani profiler'da test ediliyor, esik de orada tanimli: gerekce
# metninde ayni sabit kullanilsin ki basilan esik koddan sapmasin.
from automl.agents.profiler import ID_ORAN_ESIGI
from automl.memory.store import benzer_runlar
from automl.schemas import (ColumnDecision, DataProfile, PreprocessingPlan,
                            RunState)

SYSTEM_PROMPT = (
    "Sen bir ML preprocessing uzmanisin. Verilen veri profiline bakarak "
    "preprocessing kararlari oner."
)

# Karar esikleri: hem karsilastirmada hem gerekce metninde ayni sabit
# kullanilir, boylece basilan esik koddan sapamaz.
NULL_ESIGI = 0.5          # bu oranin ustunde null olan kolon atilir
KARDINALITE_ESIGI = 50    # bundan fazla essiz kategorili kolon atilir
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


def plan(state: RunState) -> RunState:
    "Profile bakarak preprocessing kararlarini otomatik uretir."
    p = state.profile
    if p is None:
        raise RuntimeError("plan: once profile adimi calismali")

    notes = []

    numeric_cols = []
    categorical_cols = []
    drop_cols = []
    kararlar: list[ColumnDecision] = []

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
            drop_cols.append(c.name)
            notes.append(f"{c.name}: {c.n_unique} essiz kategori, atildi")
            kararlar.append(ColumnDecision(
                name=c.name, inferred_type=c.inferred_type, decision="drop",
                reason="kardinalite çok yüksek, one-hot kodlama boyutu "
                       "patlatır ve model seyrek veriye boğulur",
                trigger=f"{c.n_unique} eşsiz kategori > {kard_esik_metni}",
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
            kararlar.append(ColumnDecision(
                name=c.name, inferred_type=c.inferred_type,
                decision="numeric",
                reason="sayısal tip, sayısal pipeline'a gider",
                trigger=f"null oranı %{c.null_ratio*100:.0f}, "
                        f"{null_esik_metni} altında",
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
        "categorical_cols, drop_cols, numeric_imputation, "
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
                + oneri.drop_cols)
    for ad in onerilen:
        if ad not in gecerli:
            print(f"   ! LLM uydurma kolon adi verdi: {ad!r}, plan reddedildi")
            return False
    if not oneri.numeric_cols and not oneri.categorical_cols:
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
    llm_karari: dict[str, str] = {}
    for ad in oneri.numeric_cols:
        llm_karari[ad] = "numeric"
    for ad in oneri.categorical_cols:
        llm_karari[ad] = "categorical"
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
