"""Test setinde degerlendirme ve feature importance."""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

from automl.agents.base import Agent
from automl.agents.modeler import BASELINE
from automl.agents.profiler import ana_metrik
from automl.schemas import DataProfile, ModelScore, RunResult, RunState

# Secilen modelin CV skoru Baseline'inkini en az bu kadar (mutlak fark)
# gecmeli. Gerekce: f1 ve r2 0-1 olceginde; 0.05'ten kucuk fark CV
# fold'lari arasindaki tipik oynaklik mertebesindedir, yani model sinif
# frekansinin (ya da ortalamanin) otesinde kayda deger bir sey ogrenmemis
# olabilir. Dengesiz veride f1_weighted ile bu fark 0.002'ye kadar iner.
BASELINE_FARK_ESIGI = 0.05


def _gini_coefficient(model, X_te, y_te) -> float | None:
    """Gini katsayisi (2*AUC - 1): kredi skorlamada yaygin metrik.
    0 = rastgele tahmin, 1 = mukemmel ayirim. Sadece ikili
    siniflandirmada anlamli oldugu icin diger durumlarda None doner.
    """
    try:
        siniflar = np.unique(y_te)
        if len(siniflar) != 2:
            return None
        if not hasattr(model, "predict_proba"):
            return None
        olasilik = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, olasilik)
        return float(2 * auc - 1)
    except Exception:
        return None


def _gini_importance(model, feat_names) -> dict[str, float]:
    """Agac-bazli Gini importance (mean decrease in impurity).

    Sadece agac modellerinde (RandomForest, GradientBoosting vb.)
    bulunur; permutation importance'a hizli bir alternatiftir.
    """
    try:
        if not hasattr(model, "feature_importances_"):
            return {}
        degerler = model.feature_importances_
        if len(degerler) != len(feat_names):
            return {}
        pairs = sorted(
            zip(feat_names, degerler),
            key=lambda x: x[1],
            reverse=True,
        )
        return {k: float(v) for k, v in pairs[:10]}
    except Exception:
        return {}


def _azinlik_etiketi(model, y, minority_class: str):
    """Profildeki azinlik sinifi adini (str) gercek etikete cevirir.

    Profil sinif adlarini str saklar, model ise orijinal tipi (int, bool...)
    gorur.
    """
    siniflar = getattr(model, "classes_", None)
    if siniflar is None:
        siniflar = np.unique(y)
    for etiket in siniflar:
        if str(etiket) == minority_class:
            return etiket
    return None


def _dengesiz_metrikler(model, X_te, y_te, y_pred,
                        p: DataProfile) -> tuple[dict[str, float],
                                                 dict[str, int]]:
    """Dengesiz veride azinlik sinifini gorunur kilan ek metrikler.

    Donen: (metrikler, azinlik confusion matrix sayimlari).
    """
    metrikler = {"f1_macro": float(f1_score(y_te, y_pred, average="macro"))}
    if p.minority_class is None:
        return metrikler, {}
    etiket = _azinlik_etiketi(model, y_te, p.minority_class)
    if etiket is None:
        return metrikler, {}

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=[etiket], zero_division=0  # type: ignore
    )
    metrikler["azinlik_precision"] = float(precision[0])  # type: ignore
    metrikler["azinlik_recall"] = float(recall[0])  # type: ignore
    metrikler["azinlik_f1"] = float(f1[0])  # type: ignore

    # PR-AUC: azinlik sinifi vs geri kalan. ROC-AUC dengesiz veride cok
    # sayidaki dogru negatif yuzunden iyimser gorunur; PR egrisi gormez.
    try:
        if hasattr(model, "predict_proba"):
            sutun = list(model.classes_).index(etiket)
            olasilik = model.predict_proba(X_te)[:, sutun]
            metrikler["pr_auc"] = float(average_precision_score(
                np.asarray(y_te) == etiket, olasilik
            ))
    except Exception as e:
        print(f"   ! pr_auc hesaplanamadi: {type(e).__name__}: {e}")

    # Confusion matrix'in azinlik satiri/sutunu: kac tanesi yakalandi?
    siniflar = list(np.unique(np.concatenate([np.asarray(y_te),
                                              np.asarray(y_pred)])))
    cm = confusion_matrix(y_te, y_pred, labels=siniflar)
    i = siniflar.index(etiket)
    yakalanan = int(cm[i, i])
    sayimlar = {
        "toplam": int(cm[i, :].sum()),
        "yakalanan": yakalanan,
        "kacirilan": int(cm[i, :].sum()) - yakalanan,
        "yanlis_alarm": int(cm[:, i].sum()) - yakalanan,
    }
    return metrikler, sayimlar


def _baseline_karsilastir(candidates: list[ModelScore], secilen: str,
                          metrik: str) -> tuple[float | None, float | None,
                                                str]:
    """Secilen modelin CV skoru Baseline'dan anlamli olcude iyi mi?

    Donen: (baseline_cv, fark, uyari). Baseline aday listesinde yoksa
    fark hesaplanamaz, (None, None, "") doner.
    """
    skorlar = {c.name: c.cv_mean for c in candidates}
    if BASELINE not in skorlar or secilen not in skorlar:
        return None, None, ""
    baseline = skorlar[BASELINE]
    fark = skorlar[secilen] - baseline

    if secilen == BASELINE:
        uyari = (f"UYARI: seçilen model Baseline'ın kendisi. Hiçbir aday "
                 f"hep aynı cevabı veren modeli {metrik} ile geçemedi.")
    elif fark < BASELINE_FARK_ESIGI:
        uyari = (f"UYARI: {secilen} Baseline'dan anlamlı ölçüde iyi değil "
                 f"(fark {fark:+.4f} < BASELINE_FARK_ESIGI "
                 f"{BASELINE_FARK_ESIGI}). Model hiçbir şey öğrenmeyen "
                 f"modelle neredeyse aynı skoru alıyor; {metrik} skoru "
                 f"yanıltıcı olabilir.")
    else:
        uyari = ""
    return baseline, fark, uyari


def evaluate(state: RunState) -> RunState:
    """Test setinde degerlendirir ve feature importance uretir."""
    p = state.profile
    if p is None:
        raise RuntimeError("profile üretilmemiş, önce profiler çalışmalı")
    if state.best_name is None:
        raise RuntimeError("model seçilmemiş, önce modeler çalışmalı")
    model = state.model
    X_te, y_te = state.X_test_t, state.y_test
    azinlik_sayimlari: dict[str, int] = {}

    y_pred = model.predict(X_te)

    if p.task_type == "regression":
        metrics = {
            "r2": float(r2_score(y_te, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "mae": float(mean_absolute_error(y_te, y_pred)),
        }
    else:
        metrics = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "f1_weighted": float(f1_score(y_te, y_pred, average="weighted")),
        }
        if p.is_imbalanced:
            ek, azinlik_sayimlari = _dengesiz_metrikler(
                model, X_te, y_te, y_pred, p
            )
            metrics.update(ek)
        gini = _gini_coefficient(model, X_te, y_te)
        if gini is not None:
            metrics["gini"] = gini
    main_metric = ana_metrik(p)

    # Feature importance: permutation, model-agnostik oldugu icin
    importance = {}
    gini_imp = {}
    try:
        feat_names = list(state.preprocessor.get_feature_names_out())
        # Dengesiz veride varsayilan skor (accuracy) azinlik sinifini
        # gormez: ozellik karistirilsa da accuracy neredeyse degismez.
        r = permutation_importance(
        model, X_te, y_te, n_repeats=5, random_state=42,
        scoring=main_metric if p.is_imbalanced else None,
        )
        pairs = sorted(
            zip(feat_names, r.importances_mean),  # type: ignore
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        importance = {k: float(v) for k, v in pairs[:10]}
        gini_imp = _gini_importance(model, feat_names)
    except Exception as e:
        print(f"   ! importance hesaplanamadi: {type(e).__name__}: {e}")

    baseline_cv, fark, uyari = _baseline_karsilastir(
        state.candidates, state.best_name, main_metric
    )

    state.result = RunResult(
        task_type=p.task_type,
        candidates=state.candidates,
        model_name=state.best_name,
        metric_name=main_metric,
        metric_value=metrics[main_metric],
        test_metrics=metrics,
        feature_importance=importance,
        gini_importance=gini_imp,
        azinlik_sayimlari=azinlik_sayimlari,
        baseline_cv=baseline_cv,
        baseline_farki=fark,
        baseline_uyarisi=uyari,
    )
    return state


class EvaluatorAgent(Agent):
    """Test setinde degerlendirir."""

    name = "evaluator"

    def run(self, state: RunState) -> RunState:
        return evaluate(state)
