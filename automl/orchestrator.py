import argparse
import time
from automl.schemas import RunState
from automl.agents.loader import LoaderAgent
from automl.agents.profiler import ProfilerAgent
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


def run(data_path: str, target: str | None = None,
        max_iterasyon: int = 3) -> RunState:
    state = RunState(data_path = data_path, target = target)

    baslangic = time.time()

    # 1) Veri hazirligi: sadece bir kez.
    for agent in HAZIRLIK:
        print(f"-> {agent.name} calisiyor...")
        state = agent(state)

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

        if _yeterli_mi(r.metric_name, r.metric_value, r.task_type):
            esik = ESIKLER.get(r.task_type)
            print(f"   esik ({esik}) gecildi, dongu durduruluyor")
            break

        if i < max_iterasyon:
            esik = ESIKLER.get(r.task_type)
            print(f"   skor esigin ({esik}) altinda, farkli strateji "
                  f"denenecek")
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
    for c in r.candidates:
        isaret = "*" if c.name == r.model_name else " "
        print(f" {isaret} {c.name:20} cv={c.cv_mean:.3f} (+/-{c.cv_std:.3f})")

    print(f"\n--- TEST SONUCU ({r.model_name}) ---")
    for k, v in r.test_metrics.items():
        print(f"  {k:12} : {v:.4f}")

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
