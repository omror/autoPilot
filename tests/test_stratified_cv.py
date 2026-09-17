"""Stratified bolme: CV fold'lari ve train/test ayrimi sinif oranini korur."""
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, StratifiedKFold

from automl.agents import modeler
from automl.agents.modeler import _cv_bolucu
from automl.agents.profiler import profile
from automl.agents.splitter import split
from automl.schemas import RunState


def _hazir_state(df: pd.DataFrame, target: str = "hedef") -> RunState:
    "Profile + split edilmis, X_train_t'si hazir state."
    st = RunState(data_path="<test>", target=target)
    st.df = df
    st = split(profile(st))
    st.X_train_t = st.X_train.to_numpy()
    return st


def _sessiz_run(yol: Path, runs_dir: Path):
    from automl.memory import logger
    from automl.orchestrator import run

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(logger, "RUNS_DIR", runs_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            return run(str(yol), "hedef")


def _az_ornekli_veri(az: int, n: int = 300) -> pd.DataFrame:
    "Iki kalabalik sinif + 'c' sinifindan sadece `az` ornek."
    rng = np.random.default_rng(0)
    y = ["a"] * n + ["b"] * n + ["c"] * az
    df = pd.DataFrame({"x1": rng.normal(size=len(y)),
                       "x2": rng.normal(size=len(y)), "hedef": y})
    df.loc[df["hedef"] == "b", "x1"] += 2
    return df


# --- CV bolucusu --------------------------------------------------------

def test_classificationda_stratified_kfold_kullanilir():
    y = pd.Series(["a", "b"] * 100)
    cv = _cv_bolucu("classification", y)
    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42


def test_regressionda_stratified_kullanilmaz():
    "Regression'da mevcut davranis: fold sayisi (int), sklearn KFold kurar."
    y = pd.Series(np.linspace(0, 1, 200))
    cv = _cv_bolucu("regression", y)
    assert not isinstance(cv, StratifiedKFold)
    assert cv == 5


def test_fold_sayisi_mantigi_korunur():
    "Kucuk veride fold duser, fold sayisi en az ornekli sinifi gecemez."
    kucuk = _cv_bolucu("classification", pd.Series(["a", "b"] * 3))
    assert isinstance(kucuk, StratifiedKFold) and kucuk.n_splits == 3

    az_sinif = _cv_bolucu("classification",
                          pd.Series(["a"] * 100 + ["b"] * 3))
    assert isinstance(az_sinif, StratifiedKFold) and az_sinif.n_splits == 3

    tek = _cv_bolucu("classification", pd.Series(["a"] * 100 + ["b"]))
    assert isinstance(tek, StratifiedKFold) and tek.n_splits == 2

    assert _cv_bolucu("regression", pd.Series([1.0, 2.0, 3.0, 4.0])) == 2


def test_dengesiz_veride_her_foldda_azinlik_ornegi_var():
    """Azinlik ornekleri verinin basinda toplu: duz KFold bunlari tek fold'a
    yigar, StratifiedKFold her fold'a dagitir."""
    y = pd.Series([1] * 10 + [0] * 990)
    X = np.zeros((len(y), 1))

    duz = [int(y.iloc[te].sum()) for _, te in KFold(5).split(X)]
    assert 0 in duz, "kurgu: duz KFold'da azinliksiz fold olmali"

    cv = _cv_bolucu("classification", y)
    assert isinstance(cv, StratifiedKFold)
    for _, te in cv.split(X, y):
        assert int(y.iloc[te].sum()) == 2


def test_train_cross_val_scorea_bolucuyu_verir(monkeypatch):
    "train() gercekten bu bolucuyu kullaniyor mu, ve regression'da degil mi?"
    verilen = []

    def sahte_cv(model, X, y, cv, scoring):
        verilen.append(cv)
        return np.array([0.5, 0.5])

    monkeypatch.setattr(modeler, "cross_val_score", sahte_cv)

    rng = np.random.default_rng(0)
    siniflama = pd.DataFrame({"x": rng.normal(size=200),
                              "hedef": ["a", "b"] * 100})
    modeler.train(_hazir_state(siniflama))
    assert verilen and all(isinstance(cv, StratifiedKFold) for cv in verilen)

    verilen.clear()
    regresyon = pd.DataFrame({"x": rng.normal(size=200),
                              "hedef": rng.normal(size=200)})
    modeler.train(_hazir_state(regresyon))
    assert verilen and all(cv == 5 for cv in verilen)


# --- Az ornekli siniflar -----------------------------------------------

@pytest.mark.parametrize("az", [2, 1])
def test_cok_az_ornekli_sinif_cokmez(tmp_path, az):
    yol = tmp_path / "az.csv"
    _az_ornekli_veri(az).to_csv(yol, index=False)
    st = _sessiz_run(yol, tmp_path / "runs")
    assert st.result is not None
    assert all("hata" not in d for d in st.gecmis_denemeler)


def test_azinlik_test_setinde_yoksa_metrikler_cokmez():
    "Tek ornekli azinlik train'e dusup test'te hic yoksa evaluator cokmemeli."
    from automl.agents.evaluator import _dengesiz_metrikler

    st = RunState(data_path="<test>", target="hedef")
    st.df = _az_ornekli_veri(1)
    p = profile(st).profile
    assert p is not None and p.minority_class == "c"

    X = np.zeros((601, 1))
    model = DummyClassifier(strategy="most_frequent").fit(X, st.df["hedef"])
    y_te = pd.Series(["a"] * 10 + ["b"] * 10)
    metrikler, sayimlar = _dengesiz_metrikler(
        model, np.zeros((20, 1)), y_te, model.predict(np.zeros((20, 1))), p)

    assert sayimlar["toplam"] == 0
    assert sayimlar["yakalanan"] == 0 and sayimlar["kacirilan"] == 0
    assert "f1_macro" in metrikler
    # Test setinde azinlik yokken recall/PR-AUC tanimsiz: 0 diye raporlanmaz.
    assert "azinlik_recall" not in metrikler
    assert "pr_auc" not in metrikler


# --- Splitter: fold mantigiyla tutarli tabakalama ----------------------

def test_tek_ornekli_sinif_tabakalamayi_kapatmaz():
    """Tek ornekli bir sinif, diger siniflarin train/test oranini bozmamali.

    Bu kurguda tabakasiz bolme 10 ornekli 'b' sinifindan test'e HIC ornek
    koymaz; tabakalama tam %20'sini (2 ornek) koyar.
    """
    y = ["a"] * 190 + ["b"] * 10 + ["c"]
    df = pd.DataFrame({"x": np.arange(len(y)), "hedef": y})
    st = RunState(data_path="<test>", target="hedef")
    st.df = df
    st = split(profile(st))
    assert int((st.y_test == "b").sum()) == 2
    assert len(st.y_test) == 41
    assert len(st.y_train) + len(st.y_test) == len(y)
