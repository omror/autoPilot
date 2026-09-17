import argparse
import textwrap
import time
from automl.schemas import (ColumnDecision, ColumnProfile, DataProfile,
                            PreprocessingPlan, RunState)
from automl.agents.loader import LoaderAgent
from automl.agents.profiler import (DENGESIZLIK_ESIGI, ProfilerAgent,
                                    ana_metrik, esik_gerekcesi)
from automl.agents.splitter import SplitterAgent
from automl.agents.planner import PlannerAgent
from automl.agents.preprocessor import PreprocessorAgent
from automl.agents.modeler import ModelerAgent
from automl.agents.evaluator import EvaluatorAgent
from automl.agents.recorder import RecorderAgent



# Veriye bagli adimlar bir kez calisir: ayni split korunmali ki
# iterasyonlar birbiriyle karsilastirilabilir olsun.
HAZIRLIK = [LoaderAgent(), ProfilerAgent(), SplitterAgent()]
# Her iterasyonda tekrar calisan adimlar.
DENEME = [PlannerAgent(), PreprocessorAgent(), ModelerAgent(),
          EvaluatorAgent()]
RECORDER = RecorderAgent()

# Bu skorun altinda kalan sonuc "yetersiz" sayilir, yeni strateji denenir.
ESIKLER = {"classification": 0.80, "regression": 0.50}

# Iterasyon sirasina gore denenecek stratejiler.
STRATEJILER = ["varsayilan", "pca_ters", "imputation_degis"]

# En iyi iterasyonu geri yuklemek icin saklanan alanlar.
_GORUNTU_ALANLARI = ("plan", "result", "model", "best_name", "candidates",
                     "preprocessor", "X_train_t", "X_test_t",
                     "strateji", "iterasyon")


def _yeterli_mi(metric_name: str, deger: float, task_type: str) -> bool:
    "Sonuc esigi gecti mi?"
    esik = ESIKLER.get(task_type)
    if esik is None:
        return True  # esigi tanimsiz task icin tek iterasyon yeter
    return deger >= esik


def _goruntu_al(state: RunState) -> dict:
    "Bir iterasyonun sonucunu geri yuklenebilir sekilde saklar."
    return {alan: getattr(state, alan) for alan in _GORUNTU_ALANLARI}


def _goruntu_yukle(state: RunState, goruntu: dict) -> None:
    "Saklanan iterasyon sonucunu state'e geri yazar."
    for alan, deger in goruntu.items():
        setattr(state, alan, deger)


# Gruplu listede en fazla kac kolon adi basilir (tam liste PLAN bolumunde).
GRUP_KOLON_LIMITI = 15


def _etiketli_yaz(etiket: str, metin: str) -> None:
    "Uzun gerekceleri terminalde tasmayacak sekilde sararak basar."
    bas = f"    {etiket} : "
    print(textwrap.fill(metin, width=76, initial_indent=bas,
                        subsequent_indent=" " * len(bas)))


def _kolon_notu(karar: ColumnDecision, c: ColumnProfile | None) -> str:
    """Gruplu listede kolon adinin yanina yazilacak dikkat cekici deger.

    Sadece bilgi veren durumda parantez acilir: sayisalda eksik veri varsa,
    kategorikte essiz deger sayisi.
    """
    if c is None:
        return karar.name
    if karar.decision == "numeric":
        if c.null_ratio > 0:
            return f"{karar.name} (null %{c.null_ratio*100:.0f})"
        return karar.name
    return f"{karar.name} ({c.n_unique} eşsiz)"


def _grup_yaz(baslik: str, islem: str, kararlar: list[ColumnDecision],
              profil: dict[str, ColumnProfile]) -> None:
    "Ayni karari alan kolonlari tek blok halinde basar."
    print(f"  {baslik} ({len(kararlar)} kolon)")
    # Ayni gruptaki kolonlarin gerekcesi normalde aynidir; farkli gerekce
    # varsa (LLM karari) hepsi ayri satirda gorunsun.
    for sebep in dict.fromkeys(k.reason for k in kararlar):
        _etiketli_yaz("sebep", sebep)
    _etiketli_yaz("işlem", islem)

    adlar = [_kolon_notu(k, profil.get(k.name))
             for k in kararlar[:GRUP_KOLON_LIMITI]]
    kalan = len(kararlar) - len(adlar)
    if kalan > 0:
        adlar.append(f"... ve {kalan} kolon daha")
    print(textwrap.fill(", ".join(adlar), width=76,
                        initial_indent="    kolonlar: ",
                        subsequent_indent="              "))


def _dengesizlik_yaz(p: DataProfile) -> None:
    """Siniflandirmada dengesizlik kararini basar.

    Dengeli veride tek satir; dengesizse dagilim, oran, azinlik sinifi,
    gecilen metrik ve NEDEN gecildigi belirgin bir blokta.
    """
    if p.task_type != "classification" or p.minority_ratio is None:
        return
    if not p.is_imbalanced:
        print(textwrap.fill(f"Dengesizlik: yok. {p.imbalance_reason}",
                            width=76, subsequent_indent="  "))
        return

    def satir(etiket: str, metin: str) -> None:
        _etiketli_yaz(f"{etiket:10}", metin)

    cizgi = "!" * 76
    print(f"\n{cizgi}")
    print("!!! SINIF DENGESIZLIGI TESPIT EDILDI")
    print(cizgi)
    # Oranlar hedefin dolu satirlari uzerinden: sayimi da oradan geri kur.
    hedef = next((c for c in p.columns if c.name == p.target), None)
    dolu = p.n_rows * (1 - (hedef.null_ratio if hedef else 0.0))
    print("  Sınıf dağılımı:")
    for sinif, oran in sorted((p.class_balance or {}).items(),
                              key=lambda x: -x[1]):
        isaret = "  <- azınlık" if sinif == p.minority_class else ""
        print(f"    {sinif[:20]:20} %{oran * 100:6.2f}  "
              f"({round(oran * dolu)} satır){isaret}")
    satir("oran", f"{p.imbalance_ratio:.1f} : 1 (çoğunluk / azınlık)")
    satir("azınlık", f"'{p.minority_class}' %{p.minority_ratio * 100:.2f} "
                     f"< DENGESIZLIK_ESIGI %{DENGESIZLIK_ESIGI * 100:g}")
    satir("metrik", f"f1_weighted -> {ana_metrik(p)} (CV, test ve "
                    f"iterasyon eşiği)")
    satir("neden", p.imbalance_reason)
    satir("eşik", esik_gerekcesi())
    satir("modeller", "havuza class_weight='balanced' varyantları eklendi "
                      "(LogisticRegression, RandomForest); mevcutlar da "
                      "kaldı, CV karşılaştırır")
    print(cizgi)


def _preprocessing_kararlari_yaz(pl: PreprocessingPlan,
                                 p: DataProfile) -> None:
    "Her kolon icin verilen karari, gerekcesini ve tetikleyen degeri basar."
    print("\n--- PREPROCESSING KARARLARI ---")
    profil = {c.name: c for c in p.columns}

    if not pl.column_decisions:
        print("  (kolon karar kaydı yok)")
    else:
        kararlar = pl.column_decisions
        atilanlar = [k for k in kararlar if k.decision == "drop"]
        sayisallar = [k for k in kararlar if k.decision == "numeric"]
        kategorikler = [k for k in kararlar if k.decision == "categorical"]
        hedefler = [k for k in kararlar if k.decision == "target"]

        # Atilanlar tek tek: her birinin sebebi farkli olabilir.
        if atilanlar:
            print(f"\nATILAN KOLONLAR ({len(atilanlar)} kolon)")
            for k in atilanlar:
                print(f"  {k.name[:20]:20} [{k.inferred_type}]  ATILDI")
                _etiketli_yaz("sebep", k.reason)
                _etiketli_yaz("tetik", k.trigger)
        else:
            print("\nATILAN KOLONLAR: yok, tum ozellikler kullanildi")

        # Tutulanlar gruplu: 30 kolonlu veride ekrani bogmasin.
        print(f"\nTUTULAN KOLONLAR "
              f"({len(sayisallar) + len(kategorikler)} kolon)")
        if sayisallar:
            _grup_yaz("sayısal pipeline",
                      f"{pl.numeric_imputation} imputation + "
                      f"{pl.scaling} scaling",
                      sayisallar, profil)
        if kategorikler:
            _grup_yaz("kategorik pipeline",
                      f"{pl.categorical_imputation} imputation + "
                      f"{pl.encoding} encoding",
                      kategorikler, profil)

        if hedefler:
            print("\nHEDEF KOLON")
            for k in hedefler:
                print(f"  {k.name[:20]:20} [{k.inferred_type}]")
                _etiketli_yaz("sebep", k.reason)
                _etiketli_yaz("tetik", k.trigger)

    # Bu veride hic kullanilmayan adimin gerekcesi basilmaz.
    atlanacak = set()
    if not pl.numeric_cols:
        atlanacak |= {"numeric_imputation", "scaling"}
    if not pl.categorical_cols:
        atlanacak |= {"categorical_imputation", "encoding"}
    gerekceler = {ad: g for ad, g in pl.step_reasons.items()
                  if ad not in atlanacak}

    if gerekceler:
        print("\nADIM GEREKÇELERİ")
        for adim, gerekce in gerekceler.items():
            print(textwrap.fill(
                gerekce, width=76,
                initial_indent=f"  {adim:22} : ",
                subsequent_indent=" " * 27))

    print("\nÖZET")
    print(f"  kolonlar: {len(pl.numeric_cols) + len(pl.categorical_cols)} "
          f"tutuldu ({len(pl.numeric_cols)} sayısal + "
          f"{len(pl.categorical_cols)} kategorik), "
          f"{len(pl.drop_cols)} atıldı")
    print(textwrap.fill(pl.pca_reason or "-", width=76,
                        initial_indent="  PCA     : ",
                        subsequent_indent=" " * 12))


def run(data_path: str, target: str | None = None,
        max_iterasyon: int = 3) -> RunState:
    state = RunState(data_path = data_path, target = target)

    baslangic = time.time()

    # 1) Veri hazirligi: sadece bir kez.
    for agent in HAZIRLIK:
        print(f"-> {agent.name} calisiyor...")
        state = agent(state)

    profil = state.profile
    if profil is None:
        raise RuntimeError("profile adimi calismadi: state.profile bos")
    # Esik her zaman ana metrik uzerinden: dengesiz veride f1_macro.
    metrik = ana_metrik(profil)
    metrik_notu = f" [{metrik}, dengesiz veri]" if profil.is_imbalanced else ""

    # 2) Self-improvement dongusu: esigi gecene kadar strateji degistir.
    en_iyi: dict | None = None
    for i in range(1, max_iterasyon + 1):
        state.iterasyon = i
        state.strateji = STRATEJILER[min(i - 1, len(STRATEJILER) - 1)]
        print(f"\n=== ITERASYON {i}/{max_iterasyon} "
              f"(strateji: {state.strateji}) ===")

        try:
            for agent in DENEME:
                print(f"-> {agent.name} calisiyor...")
                state = agent(state)
        except Exception as e:
            print(f"   ! iterasyon {i} basarisiz: {type(e).__name__}: {e}")
            state.gecmis_denemeler.append({
                "iterasyon": i,
                "strateji": state.strateji,
                "hata": f"{type(e).__name__}: {e}",
            })
            continue

        r = state.result
        pl = state.plan
        if r is None or pl is None:
            continue

        state.gecmis_denemeler.append({
            "iterasyon": i,
            "strateji": state.strateji,
            "model_name": r.model_name,
            "metric_name": r.metric_name,
            "metric_value": r.metric_value,
            "use_pca": pl.use_pca,
            "n_components": pl.n_components,
            "numeric_imputation": pl.numeric_imputation,
            "categorical_imputation": pl.categorical_imputation,
        })

        print(f"   iterasyon {i} sonucu: {r.metric_name}="
              f"{r.metric_value:.4f}  (model: {r.model_name})")

        if en_iyi is None or r.metric_value > en_iyi["result"].metric_value:
            en_iyi = _goruntu_al(state)
            print("   -> simdiye kadarki en iyi sonuc")

        if _yeterli_mi(metrik, r.test_metrics[metrik], r.task_type):
            esik = ESIKLER.get(r.task_type)
            print(f"   esik ({esik}){metrik_notu} gecildi, dongu durduruluyor")
            break

        if i < max_iterasyon:
            esik = ESIKLER.get(r.task_type)
            print(f"   skor esigin ({esik}){metrik_notu} altinda, farkli "
                  f"strateji denenecek")
        else:
            print(f"   max iterasyona ({max_iterasyon}) ulasildi, duruluyor")

    # 3) Son sonucu degil, EN IYI sonucu raporla.
    if en_iyi is not None:
        _goruntu_yukle(state, en_iyi)
        print(f"\nEn iyi iterasyon: {state.iterasyon} "
              f"(strateji: {state.strateji})")
    else:
        raise RuntimeError("hicbir iterasyon sonuc uretemedi")

    state.sure_sn = time.time() - baslangic
    print(f"-> {RECORDER.name} calisiyor...")
    state = RECORDER(state)

    p = state.profile
    if p is None:
        raise RuntimeError("profile adimi calismadi: state.profile bos")

    print("\n--- PROFILE ---")
    print(f"Satır: {p.n_rows} Kolon: {p.n_cols}")
    print(f"Hedef: {p.target} Task: {p.task_type}")
    print(f"Sınıf dengesi: {p.class_balance}")
    _dengesizlik_yaz(p)
    print("\n Kolonlar:")
    for c in p.columns:
        print(f" {c.name:15}, {c.dtype:10} -> {c.inferred_type:12}"
              f"(essiz: {c.n_unique}, bos: {c.null_ratio:.0%})")
    
    # Betimsel istatistik tablosu
    sayisal_kolonlar = [c for c in p.columns if c.inferred_type == "numeric"]
    if sayisal_kolonlar:
        print("\n--- ISTATISTIK (sayisal) ---")
        print(f"  {'kolon':22} {'ort':>10} {'std':>10} {'min':>10} "
              f"{'medyan':>10} {'max':>10} {'carpiklik':>10}")
        for c in sayisal_kolonlar[:15]:
            # Tamamen bos kolonda istatistik yok, None gelir: "-" yaz.
            def _f(x) -> str:
                return f"{x:>10.2f}" if x is not None else f"{'-':>10}"
            print(f"  {c.name[:22]:22} {_f(c.mean)} {_f(c.std)} "
                  f"{_f(c.min)} {_f(c.median)} {_f(c.max)} "
                  f"{_f(c.skew)}")
        if len(sayisal_kolonlar) > 15:
            print(f"  ... ve {len(sayisal_kolonlar) - 15} kolon daha")

    kategorik_kolonlar = [c for c in p.columns
                          if c.inferred_type == "categorical"]
    if kategorik_kolonlar:
        print("\n--- ISTATISTIK (kategorik) ---")
        for c in kategorik_kolonlar:
            oran = f"%{c.top_ratio * 100:.1f}" if c.top_ratio is not None else "-"
            print(f"  {c.name[:22]:22} en sik: {c.top_value} "
                  f"({oran}), {c.n_unique} essiz")

    if p.high_correlations:
        print("\n--- YUKSEK KORELASYONLAR (|r| >= 0.85) ---")
        for a, b, r_deger in p.high_correlations:
            print(f"  {a[:25]:25} <-> {b[:25]:25} r={r_deger:+.3f}")


    pl = state.plan
    if pl is None:
        raise RuntimeError("plan adimi calismadi: state.plan bos")

    print("\n--- PLAN ---")
    print(f"Sayisal   : {pl.numeric_cols}")
    print(f"Kategorik : {pl.categorical_cols}")
    print(f"Atilan    : {pl.drop_cols}")
    print(f"Imputation: {pl.numeric_imputation} / {pl.categorical_imputation}")
    print(f"Scaling   : {pl.scaling}   Encoding: {pl.encoding}")
    print(f"PCA       : {pl.use_pca} ({pl.n_components})")
    for n in pl.notes:
        print(f"  not: {n}")

    _preprocessing_kararlari_yaz(pl, p)

    print("\n--- PREPROCESS ---")
    print(f"X_train: {state.X_train.shape} -> {state.X_train_t.shape}")
    print(f"X_test : {state.X_test.shape} -> {state.X_test_t.shape}")

    r = state.result
    if r is None:
        raise RuntimeError("sonuç üretilmemiş")
    if state.gecmis_denemeler:
        print("\n--- ITERASYONLAR ---")
        for d in state.gecmis_denemeler:
            isaret = "*" if d["iterasyon"] == state.iterasyon else " "
            if "hata" in d:
                print(f" {isaret} {d['iterasyon']}. {d['strateji']:18} "
                      f"HATA: {d['hata']}")
                continue
            print(f" {isaret} {d['iterasyon']}. {d['strateji']:18} "
                  f"{d['metric_name']}={d['metric_value']:.4f}  "
                  f"model={d['model_name']:18} "
                  f"PCA={str(d['use_pca']):5} "
                  f"imp={d['numeric_imputation']}")
        print("  (* = raporlanan en iyi iterasyon)")

    print("\n--- MODEL KARSILASTIRMA ---")
    genislik = max([20] + [len(c.name) for c in r.candidates])
    for c in r.candidates:
        isaret = "*" if c.name == r.model_name else " "
        print(f" {isaret} {c.name:{genislik}} cv={c.cv_mean:.3f} "
              f"(+/-{c.cv_std:.3f})")
    if r.baseline_farki is not None and r.baseline_cv is not None:
        print(f"  Baseline farkı ({r.metric_name}, cv): {r.model_name} "
              f"{r.baseline_cv + r.baseline_farki:.4f} - Baseline "
              f"{r.baseline_cv:.4f} = {r.baseline_farki:+.4f}")
    else:
        print("  Baseline farkı: hesaplanamadı (Baseline aday listesinde yok)")
    if r.baseline_uyarisi:
        print("  " + "!" * 74)
        print(textwrap.fill(r.baseline_uyarisi, width=76,
                            initial_indent="  ! ", subsequent_indent="  ! "))
        print("  " + "!" * 74)

    print(f"\n--- TEST SONUCU ({r.model_name}) ---")
    genislik = max([12] + [len(k) for k in r.test_metrics])
    for k, v in r.test_metrics.items():
        ana = "  <- ana metrik" if p.is_imbalanced and k == r.metric_name else ""
        print(f"  {k:{genislik}} : {v:.4f}{ana}")
    if r.azinlik_sayimlari:
        s = r.azinlik_sayimlari
        print(textwrap.fill(
            f"azınlık sınıfı '{p.minority_class}' (test setinde "
            f"{s['toplam']} örnek): {s['yakalanan']} yakalandı, "
            f"{s['kacirilan']} kaçırıldı, {s['yanlis_alarm']} yanlış alarm",
            width=76, initial_indent="  ", subsequent_indent="    "))
    if p.is_imbalanced and "pr_auc" in r.test_metrics:
        print(textwrap.fill(
            f"not: pr_auc'ta rastgele tahmin seviyesi azınlık oranıdır "
            f"(≈{p.minority_ratio:.4f}). gini ROC tabanlıdır, dengesiz "
            f"veride çok sayıdaki doğru negatif yüzünden iyimser görünür.",
            width=76, initial_indent="  ", subsequent_indent="  "))

    if r.feature_importance:
        print("\n--- FEATURE IMPORTANCE ---")
        for k, v in r.feature_importance.items():
            print(f"  {k:25} {v:+.4f}")

    if r.gini_importance:
        print("\n--- GINI IMPORTANCE (agac-bazli) ---")
        for k, v in r.gini_importance.items():
            print(f"  {k:25} {v:.4f}")
        print("  not: secilen model agac-bazli oldugu icin hesaplandi")
    elif p.task_type != "regression":
        print("  not: secilen model agac-bazli degil, "
              "Gini importance hesaplanamaz")
    return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default = None)
    parser.add_argument("--max-iterasyon", type=int, default=3)
    args = parser.parse_args()
    run(args.data, args.target, args.max_iterasyon)
