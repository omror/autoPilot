"""Veriyi tanir: kolon tipleri, hedef kolon ve gorev tipi."""
import numpy as np
import pandas as pd
from automl.agents.base import Agent
from automl.schemas import (
    ColumnProfile,
    DataProfile,
    InferredType,
    RunState,
    TaskType,
)


def _detect_type(s: pd.Series, n_rows:int) -> InferredType:
    "Bir kolonun gerçek tipini otomatik olarak tespit eder."

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        n_unique = s.nunique(dropna=True)
        if n_unique > 50 and n_unique / max(n_rows, 1) > 0.5:
            return "text"
        return "categorical"

    if pd.api.types.is_bool_dtype(s):
        return "categorical"

    if pd.api.types.is_numeric_dtype(s):
        clean = s.dropna()
        if len(clean) == 0:
            return "numeric"

        # Kural 1: Ondalıklı değer varsa kesin sayısal
        if not clean.mod(1).eq(0).all():
            return "numeric"

        #Kural 2: eşik veri boyutuna göre otomatik hesaplanır.
        esik = max(2, min(20, int(n_rows * 0.05)))
        return "categorical" if clean.nunique() <= esik else "numeric"

    return "categorical"

def _numeric_stats(s: pd.Series) -> dict:
    """Sayisal kolon icin betimsel istatistikler."""
    clean = s.dropna()
    if len(clean) == 0:
        return {}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,  # type: ignore
        "min": float(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "skew": float(clean.skew()) if len(clean) > 2 else 0.0,  # type: ignore
    }


def _categorical_stats(s: pd.Series) -> dict:
    """Kategorik kolon icin en sik deger ve orani."""
    clean = s.dropna()
    if len(clean) == 0:
        return {}
    vc = clean.value_counts(normalize=True)
    return {
        "top_value": str(vc.index[0]),
        "top_ratio": float(vc.iloc[0]),
    }


def _high_correlations(df: pd.DataFrame, cols: list[str],
                       esik: float = 0.85) -> list[tuple[str, str, float]]:
    """Yuksek korelasyonlu sayisal kolon ciftlerini bulur."""
    if len(cols) < 2:
        return []
    try:
        corr = df[cols].corr()  # type: ignore
    except Exception:
        return []

    ciftler = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            deger = corr.loc[a, b]
            if pd.notna(deger) and abs(float(deger)) >= esik:
                ciftler.append((str(a), str(b), round(float(deger), 3)))

    ciftler.sort(key=lambda x: abs(x[2]), reverse=True)
    return ciftler[:15]

def profile(state: RunState) -> RunState:

    df = state.df
    n_rows = len(df)

    columns = []
    for name in df.columns:
        s = df[name]
        tip = _detect_type(s, n_rows)

        # Tipe gore ek istatistikler
        ekstra = {}
        if tip == "numeric":
            ekstra = _numeric_stats(s)
        elif tip == "categorical":
            ekstra = _categorical_stats(s)

        columns.append(
            ColumnProfile(
                name=name,
                dtype=str(s.dtype),
                inferred_type=tip,
                n_unique=int(s.nunique(dropna=True)),
                null_ratio=float(s.isna().mean()),
                **ekstra,
            )
        )

    target = state.target
    if target is None and len(df.columns) > 0:
        target = df.columns[-1]

    class_balance = None
    task_type: TaskType
    if target is None:
        task_type = "clustering"
    else:
        hedef_profil = next((c for c in columns if c.name == target), None)
        if hedef_profil is None:
            raise ValueError(
                f"Hedef kolon '{target}' veri setinde yok. "
                f"Mevcut kolonlar: {list(df.columns)}"
            )
        if hedef_profil.inferred_type == "numeric":
            task_type = "regression"
        else:
            task_type = "classification"
            oranlar = df[target].value_counts(normalize=True)
            class_balance = {str(k): float(v) for k, v in oranlar.items()}

        # Sayisal kolonlar arasi yuksek korelasyonlar (hedef haric)
    sayisal = [c.name for c in columns
               if c.inferred_type == "numeric" and c.name != target]
    korelasyonlar = _high_correlations(df, sayisal)

    state.profile = DataProfile(
        n_rows = n_rows,
        n_cols = len(df.columns),
        target = target,
        task_type = task_type,
        columns = columns,
        class_balance=class_balance,
        high_correlations=korelasyonlar,
    )

    return state


class ProfilerAgent(Agent):
    """Veriyi profiller: kolon tipleri, hedef, gorev tipi."""

    name = "profiler"

    def run(self, state: RunState) -> RunState:
        return profile(state)
