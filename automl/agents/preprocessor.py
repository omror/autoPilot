"""Planı sklearn Pipeline'a cevirir ve uygular."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from automl.agents.base import Agent
from automl.schemas import PreprocessingPlan, RunState

# Yuksek kardinaliteli kolonda train'deki orani bunun altinda kalan kategori
# nadir sayilir ve tek kategoride toplanir. Planner karar verirken de ayni
# sabiti kullanir (encoder'in parametresi oldugu icin burada tanimli).
NADIR_KATEGORI_ESIGI = 0.01


def _make_onehot() -> OneHotEncoder:
    "Yogun (dense) matris ureten one-hot encoder. sklearn >= 1.2 gerekir."
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def _make_nadir_onehot() -> OneHotEncoder:
    """Nadir kategorileri tek kategoride toplayan one-hot encoder.

    Train'de orani NADIR_KATEGORI_ESIGI altinda kalan kategoriler fit
    sirasinda "infrequent_sklearn" kategorisinde birlesir. Liste SADECE
    fit edilen veriden (train) ogrenilir; test'te gorulmeyen kategori de
    bu gruba duser.
    """
    return OneHotEncoder(handle_unknown="infrequent_if_exist",
                         min_frequency=NADIR_KATEGORI_ESIGI,
                         sparse_output=False)


def _iki_boyutlu(X) -> np.ndarray:
    "DataFrame / 1B / 2B girdiyi 2B object dizisine cevirir."
    dizi = np.asarray(X, dtype=object)
    return dizi.reshape(-1, 1) if dizi.ndim == 1 else dizi


class FrekansKodlayici(OneToOneFeatureMixin, TransformerMixin,
                       BaseEstimator):
    """Her kategoriyi train'deki gorulme orani ile degistirir.

    Oran = kategorinin train'de gecme sayisi / train satir sayisi. Kolon
    basina tek sayisal kolon uretir, one-hot patlamasi olmaz. Oranlar
    SADECE fit'te ogrenilir; train'de gorulmeyen kategori 0 alir.
    """

    def fit(self, X, y=None):
        X = _iki_boyutlu(X)
        self.n_features_in_ = X.shape[1]
        self.frekanslar_ = [
            pd.Series(X[:, j]).value_counts(normalize=True).to_dict()
            for j in range(X.shape[1])
        ]
        return self

    def transform(self, X):
        check_is_fitted(self, "frekanslar_")
        X = _iki_boyutlu(X)
        return np.column_stack([
            pd.Series(X[:, j]).map(self.frekanslar_[j])
            .fillna(0.0).to_numpy(dtype=float)
            for j in range(X.shape[1])
        ])


def pipeline_kur(pl: PreprocessingPlan) -> ColumnTransformer | Pipeline:
    """Plani fit edilmemis sklearn on isleme adimina cevirir.

    Hem preprocess hem planner'in katki olcumu (CV icinde) bunu kullanir,
    boylece olculen on isleme ile uygulanan birebir aynidir.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=pl.numeric_imputation)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=pl.categorical_imputation)),
        ("encoder", _make_onehot()),
    ])

    # Yuksek kardinaliteli kategorikler: nadir kategori listesi ve frekanslar
    # encoder'in fit'inde, yani sadece train'den ogrenilir.
    nadir_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=pl.categorical_imputation)),
        ("encoder", _make_nadir_onehot()),
    ])
    frekans_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=pl.categorical_imputation)),
        ("encoder", FrekansKodlayici()),
        ("scaler", StandardScaler()),
    ])

    transformers = []
    if pl.numeric_cols:
        transformers.append(("num", numeric_pipe, pl.numeric_cols))
    if pl.categorical_cols:
        transformers.append(("cat", categorical_pipe, pl.categorical_cols))
    if pl.nadir_toplama_cols:
        transformers.append(("nadir", nadir_pipe, pl.nadir_toplama_cols))
    if pl.frekans_cols:
        transformers.append(("frekans", frekans_pipe, pl.frekans_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")

    if pl.use_pca:
        return Pipeline([
            ("ct", ct),
            ("pca", PCA(n_components=pl.n_components, random_state=42)),
        ])
    return ct


def preprocess(state: RunState) -> RunState:
    "Planı sklearn Pipeline'a çevirir. fit SADECE train üzerinde."
    pl = state.plan
    if pl is None:
        raise RuntimeError("preprocess: once plan adimi calismali")

    preprocessor = pipeline_kur(pl)

    # KRITIK: fit sadece train'de, test'e sadece transform
    state.X_train_t = preprocessor.fit_transform(state.X_train)
    state.X_test_t = preprocessor.transform(state.X_test)
    state.preprocessor = preprocessor
    return state


class PreprocessorAgent(Agent):
    """Plani Pipeline'a cevirir ve uygular."""

    name = "preprocessor"

    def run(self, state: RunState) -> RunState:
        return preprocess(state)
