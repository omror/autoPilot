"""Sinif dengesizligi: tespit, metrik secimi, model havuzu, baseline farki."""
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.agents import profiler
from automl.agents.evaluator import BASELINE_FARK_ESIGI, _baseline_karsilastir
from automl.agents.modeler import _candidate_models
from automl.agents.profiler import DENGESIZLIK_ESIGI, ana_metrik, profile
from automl.schemas import ModelScore, RunState

IRIS = "data/iris.csv"
DENGELI_HAVUZ = ["Baseline", "LogisticRegression", "RandomForest",
                 "GradientBoosting"]


def _profil(df: pd.DataFrame, target: str):
    st = RunState(data_path="<test>", target=target)
    st.df = df
    p = profile(st).profile
    assert p is not None
    return p


def _siniflar(sayimlar: dict, n_ozellik: int = 2) -> pd.DataFrame:
    "Verilen sinif sayimlariyla rastgele ozellikli bir veri uretir."
    y = [s for s, n in sayimlar.items() for _ in range(n)]
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"x{i}": rng.normal(size=len(y))
                       for i in range(n_ozellik)})
    df["hedef"] = y
    return df


def _sessiz_run(yol: str, target: str | None, runs_dir: Path):
    "Orchestrator'i ciktisiz ve runs/ klasorunu kirletmeden calistirir."
    from automl.memory import logger
    from automl.orchestrator import run

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(logger, "RUNS_DIR", runs_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            return run(yol, target)


# --- 1) Dengesizlik tespiti ---------------------------------------------

def test_ikili_dengesiz_veri_tespit_edilir():
    "%5 azinlik -> dengesiz; azinlik sinifi ve oranlar dogru hesaplanmali."
    p = _profil(_siniflar({0: 950, 1: 50}), "hedef")
    assert p.is_imbalanced
    assert p.minority_class == "1"
    assert p.minority_ratio == pytest.approx(0.05)
    assert p.imbalance_ratio == pytest.approx(19.0)
    assert "f1_macro" in p.imbalance_reason


def test_cok_sinifli_veride_en_kucuk_sinifa_bakilir():
    "Cok sinifli veride de en kucuk sinif belirleyici olmali."
    p = _profil(_siniflar({"a": 600, "b": 350, "c": 50}), "hedef")
    assert p.is_imbalanced
    assert p.minority_class == "c"
    assert p.minority_ratio == pytest.approx(0.05)
    assert p.imbalance_ratio == pytest.approx(12.0)


def test_dengeli_veri_dengesiz_sayilmaz():
    "Esit dagilimli uc sinif dengeli kalmali; bilgi alanlari yine dolu."
    p = _profil(_siniflar({"a": 100, "b": 100, "c": 100}), "hedef")
    assert not p.is_imbalanced
    assert p.minority_ratio == pytest.approx(1 / 3)
    assert p.imbalance_ratio == pytest.approx(1.0)


def test_titanic_gibi_hafif_dengesizlik_dengeli_sayilir():
    "%38 azinlik esigin cok ustunde: metrik degismemeli."
    p = _profil(_siniflar({0: 616, 1: 384}), "hedef")
    assert not p.is_imbalanced
    assert ana_metrik(p) == "f1_weighted"


def test_esik_siniri_kesin_kucuktur():
    "Oran tam esige esitse dengeli, bir ornek eksikse dengesiz."
    n = 1000
    tam = round(n * DENGESIZLIK_ESIGI)
    esitte = _profil(_siniflar({0: n - tam, 1: tam}), "hedef")
    altinda = _profil(_siniflar({0: n - tam + 1, 1: tam - 1}), "hedef")
    assert not esitte.is_imbalanced
    assert altinda.is_imbalanced


def test_regression_dengesizlik_alanlari_bos():
    "Regression'da sinif yok: dengesizlik alanlari bos kalmali."
    df = pd.DataFrame({"x": range(200),
                       "fiyat": [i * 1.7 + 0.3 for i in range(200)]})
    p = _profil(df, "fiyat")
    assert p.task_type == "regression"
    assert not p.is_imbalanced
    assert p.minority_class is None
    assert p.minority_ratio is None
    assert p.imbalance_ratio is None


def test_gerekce_metinleri_esik_sabitini_kullanir(monkeypatch):
    "Esik degisirse gerekce metni de ayni sabitten guncellenmeli."
    monkeypatch.setattr(profiler, "DENGESIZLIK_ESIGI", 0.10)
    p = _profil(_siniflar({0: 880, 1: 120}), "hedef")   # %12
    assert not p.is_imbalanced
    assert "%10" in p.imbalance_reason
    assert "%10" in profiler.esik_gerekcesi()


# --- 2) Metrik secimi ve model havuzu -----------------------------------

def test_ana_metrik_sadece_dengesiz_veride_degisir():
    dengeli = _profil(_siniflar({"a": 500, "b": 500}), "hedef")
    dengesiz = _profil(_siniflar({"a": 950, "b": 50}), "hedef")
    regresyon = _profil(pd.DataFrame({"x": range(100),
                                      "y": [i * 0.3 for i in range(100)]}),
                        "y")
    assert ana_metrik(dengeli) == "f1_weighted"
    assert ana_metrik(dengesiz) == "f1_macro"
    assert ana_metrik(regresyon) == "r2"


def test_dengesiz_veride_class_weight_varyantlari_eklenir():
    "Balanced varyantlar eklenmeli, mevcut modeller de havuzda kalmali."
    havuz = _candidate_models("classification", 1000, dengesiz=True)
    assert list(havuz)[:4] == DENGELI_HAVUZ
    assert havuz["LogisticRegression_balanced"].class_weight == "balanced"
    assert havuz["RandomForest_balanced"].class_weight == "balanced"
    assert havuz["LogisticRegression"].class_weight is None
    assert havuz["RandomForest"].class_weight is None


def test_dengeli_veride_havuz_degismez():
    assert list(_candidate_models("classification", 1000)) == DENGELI_HAVUZ
    assert list(_candidate_models("classification", 1000,
                                  dengesiz=False)) == DENGELI_HAVUZ
    # Regression'da dengesizlik bayragi havuzu etkilememeli.
    assert (list(_candidate_models("regression", 1000, dengesiz=True))
            == list(_candidate_models("regression", 1000)))


# --- 3) Baseline farki uyarisi ------------------------------------------

def _skor(ad: str, cv: float) -> ModelScore:
    return ModelScore(name=ad, cv_mean=cv, cv_std=0.0)


def test_baseline_farki_kucukse_uyari_verilir():
    "Fraud senaryosu: f1_weighted'da model baseline'dan sadece 0.002 iyi."
    adaylar = [_skor("Baseline", 0.997), _skor("RandomForest", 0.999)]
    baseline, fark, uyari = _baseline_karsilastir(adaylar, "RandomForest",
                                                  "f1_weighted")
    assert baseline == pytest.approx(0.997)
    assert fark == pytest.approx(0.002)
    assert "BASELINE_FARK_ESIGI" in uyari
    assert str(BASELINE_FARK_ESIGI) in uyari


def test_baseline_farki_buyukse_uyari_yok():
    adaylar = [_skor("Baseline", 0.49), _skor("RandomForest", 0.91)]
    _, fark, uyari = _baseline_karsilastir(adaylar, "RandomForest", "f1_macro")
    assert fark == pytest.approx(0.42)
    assert uyari == ""


def test_secilen_model_baseline_ise_uyari_verilir():
    adaylar = [_skor("Baseline", 0.5), _skor("LogisticRegression", 0.4)]
    _, fark, uyari = _baseline_karsilastir(adaylar, "Baseline", "f1_macro")
    assert fark == 0
    assert "Baseline'ın kendisi" in uyari


def test_baseline_yoksa_fark_hesaplanmaz():
    adaylar = [_skor("RandomForest", 0.9)]
    assert _baseline_karsilastir(adaylar, "RandomForest", "f1_macro") == (
        None, None, "")


# --- Uctan uca ----------------------------------------------------------

@pytest.fixture(scope="module")
def dengesiz_state(tmp_path_factory):
    "%5 azinlikli, ogrenilebilir sinyali olan veriyle tam boru hatti."
    klasor = tmp_path_factory.mktemp("dengesiz")
    rng = np.random.default_rng(42)
    n = 2000
    y = (rng.random(n) < 0.05).astype(int)
    df = pd.DataFrame({
        "a": rng.normal(size=n) + 4 * y,
        "b": rng.normal(size=n),
        "c": rng.normal(size=n) - 3 * y,
        "hedef": y,
    })
    yol = klasor / "dengesiz.csv"
    df.to_csv(yol, index=False)
    return _sessiz_run(str(yol), "hedef", klasor / "runs")


def test_uctan_uca_dengesiz_veride_f1_macro(dengesiz_state):
    r = dengesiz_state.result
    assert dengesiz_state.profile.is_imbalanced
    assert r.metric_name == "f1_macro"
    assert r.metric_value == r.test_metrics["f1_macro"]
    for d in dengesiz_state.gecmis_denemeler:
        assert d["metric_name"] == "f1_macro"


def test_uctan_uca_dengesiz_veride_azinlik_metrikleri(dengesiz_state):
    r = dengesiz_state.result
    for k in ("f1_weighted", "f1_macro", "azinlik_precision",
              "azinlik_recall", "azinlik_f1", "pr_auc"):
        assert k in r.test_metrics, k

    s = r.azinlik_sayimlari
    assert s["toplam"] == int((dengesiz_state.y_test == 1).sum())
    assert s["yakalanan"] + s["kacirilan"] == s["toplam"]
    assert r.test_metrics["azinlik_recall"] == pytest.approx(
        s["yakalanan"] / s["toplam"])


def test_uctan_uca_dengesiz_veride_balanced_varyantlar_denendi(dengesiz_state):
    adlar = [c.name for c in dengesiz_state.result.candidates]
    assert adlar == DENGELI_HAVUZ + ["LogisticRegression_balanced",
                                     "RandomForest_balanced"]
    assert dengesiz_state.result.baseline_farki is not None


@pytest.fixture(scope="module")
def iris_state(tmp_path_factory):
    if not Path(IRIS).exists():
        pytest.skip("data/iris.csv yok")
    return _sessiz_run(IRIS, None, tmp_path_factory.mktemp("iris") / "runs")


def test_dengeli_veride_metrik_ve_skor_degismez(iris_state):
    "Dengeli veride f1_weighted, eski havuz, eski metrikler ve eski skor."
    p, r = iris_state.profile, iris_state.result
    assert not p.is_imbalanced
    assert r.metric_name == "f1_weighted"
    assert r.metric_value == pytest.approx(0.9333, abs=5e-5)
    assert [c.name for c in r.candidates] == DENGELI_HAVUZ
    assert set(r.test_metrics) == {"accuracy", "f1_weighted"}
    assert r.azinlik_sayimlari == {}
    # Baseline farki dengeli veride de raporlanir, iris'te uyari gerekmez.
    assert r.baseline_farki is not None
    assert r.baseline_farki >= BASELINE_FARK_ESIGI
    assert r.baseline_uyarisi == ""
