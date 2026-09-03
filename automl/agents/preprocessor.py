"""Plani sklearn Pipeline'a cevirir ve uygular."""
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automl.agents.base import Agent
from automl.schemas import RunState


def _make_onehot() -> OneHotEncoder:
    "Yogun (dense) matris ureten one-hot encoder. sklearn >= 1.2 gerekir."
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def preprocess(state: RunState) -> RunState:
    "Planı sklearn Pipeline'a çevirir. fit SADECE train üzerinde."
    pl = state.plan
    if pl is None:
        raise RuntimeError("preprocess: once plan adimi calismali")

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=pl.numeric_imputation)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=pl.categorical_imputation)),
        ("encoder", _make_onehot()),
    ])

    transformers = []
    if pl.numeric_cols:
        transformers.append(("num", numeric_pipe, pl.numeric_cols))
    if pl.categorical_cols:
        transformers.append(("cat", categorical_pipe, pl.categorical_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")

    if pl.use_pca:
        preprocessor = Pipeline([
            ("ct", ct),
            ("pca", PCA(n_components=pl.n_components, random_state=42)),
        ])
    else:
        preprocessor = ct

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
