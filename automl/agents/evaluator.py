"""Test setinde degerlendirme ve feature importance."""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from automl.agents.base import Agent
from automl.schemas import RunResult, RunState


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


def evaluate(state: RunState) -> RunState:
    """Test setinde degerlendirir ve feature importance uretir."""
    p = state.profile
    if p is None:
        raise RuntimeError("profile üretilmemiş, önce profiler çalışmalı")
    if state.best_name is None:
        raise RuntimeError("model seçilmemiş, önce modeler çalışmalı")
    model = state.model
    X_te, y_te = state.X_test_t, state.y_test

    y_pred = model.predict(X_te)

    if p.task_type == "regression":
        metrics = {
            "r2": float(r2_score(y_te, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "mae": float(mean_absolute_error(y_te, y_pred)),
        }
        main_metric = "r2"
    else:
        metrics = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "f1_weighted": float(f1_score(y_te, y_pred, average="weighted")),
        }
        gini = _gini_coefficient(model, X_te, y_te)
        if gini is not None:
            metrics["gini"] = gini
        main_metric = "f1_weighted"

    # Feature importance: permutation, model-agnostik oldugu icin
    importance = {}
    gini_imp = {}
    try:
        feat_names = list(state.preprocessor.get_feature_names_out())
        r = permutation_importance(
        model, X_te, y_te, n_repeats=5, random_state=42
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

    state.result = RunResult(
        task_type=p.task_type,
        candidates=state.candidates,
        model_name=state.best_name,
        metric_name=main_metric,
        metric_value=metrics[main_metric],
        test_metrics=metrics,
        feature_importance=importance,
        gini_importance=gini_imp,
    )
    return state


class EvaluatorAgent(Agent):
    """Test setinde degerlendirir."""

    name = "evaluator"

    def run(self, state: RunState) -> RunState:
        return evaluate(state)
