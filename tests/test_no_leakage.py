"""Veri sizintisi onlemi: EN KRITIK test dosyasi.

Preprocessor SADECE train uzerinde fit edilmeli. Test setinin dagilimi
scaler'a sizarsa model gercekte olduğundan iyi gorunur ve rapor edilen
skor yalan olur.
"""
import numpy as np
import pandas as pd

from automl.agents.preprocessor import preprocess
from automl.schemas import PreprocessingPlan, RunState


def _scaler_al(preprocessor):
    "ColumnTransformer icindeki sayisal StandardScaler'i cikarir."
    return preprocessor.named_transformers_["num"].named_steps["scaler"]


def test_scaler_test_setinden_etkilenmiyor():
    """Test setine asiri uc degerler koy: scaler onlari GORMEMELI.

    fit test setini de gorseydi ortalama 49.5'ten milyonlara kayardi.
    """
    X_train = pd.DataFrame({"olcum": np.arange(100, dtype=float)})
    # Test seti asiri uc: egitimdekinin ~20.000 katı
    X_test = pd.DataFrame({"olcum": [1_000_000.0] * 25})

    st = RunState(data_path="<test>")
    st.X_train, st.X_test = X_train, X_test
    st.plan = PreprocessingPlan(numeric_cols=["olcum"])

    st = preprocess(st)
    scaler = _scaler_al(st.preprocessor)

    # Sadece train'in istatistikleri ogrenilmis olmali
    assert scaler.mean_[0] == 49.5, "scaler test setinin ortalamasini gormus"
    assert np.isclose(scaler.scale_[0], X_train["olcum"].std(ddof=0))

    # Uc degerler devasa z-skoruna donusmeli: test'e sadece transform
    # uygulandiginin kaniti
    assert st.X_test_t.min() > 1000


def test_test_setine_sadece_transform_uygulaniyor():
    "Ayni scaler ile donusen train ortalamasi ~0, test ise kaymis olmali."
    X_train = pd.DataFrame({"a": np.arange(200, dtype=float)})
    X_test = pd.DataFrame({"a": np.arange(500, 700, dtype=float)})

    st = RunState(data_path="<test>")
    st.X_train, st.X_test = X_train, X_test
    st.plan = PreprocessingPlan(numeric_cols=["a"])
    st = preprocess(st)

    assert abs(float(np.mean(st.X_train_t))) < 1e-9   # train merkezlenmis
    assert float(np.mean(st.X_test_t)) > 5            # test kaymis kalmis


def test_bos_deger_imputation_da_traindan_ogreniliyor():
    "Eksik deger doldurma degeri de sadece train'den hesaplanmali."
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X_test = pd.DataFrame({"a": [np.nan, 999_999.0]})

    st = RunState(data_path="<test>")
    st.X_train, st.X_test = X_train, X_test
    st.plan = PreprocessingPlan(numeric_cols=["a"])
    st = preprocess(st)

    imputer = st.preprocessor.named_transformers_["num"].named_steps["imputer"]
    # train medyani 3.0; test'teki 999999 bu degeri etkilememeli
    assert imputer.statistics_[0] == 3.0


def test_split_preprocessten_once_calisiyor():
    """Pipeline sirasi yapisal olarak dogru mu?

    splitter, preprocessor'dan ONCE gelmezse fit tum veriyi gorur.
    """
    from automl.orchestrator import DENEME, HAZIRLIK

    sira = [a.name for a in HAZIRLIK] + [a.name for a in DENEME]
    assert sira.index("splitter") < sira.index("preprocessor")
    assert sira.index("preprocessor") < sira.index("modeler")
    assert sira.index("modeler") < sira.index("evaluator")


def test_uctan_uca_scaler_sadece_traini_gormus():
    """Gercek veride: scaler'in ortalamasi X_train'e uymali, df'e degil.

    Tum veri uzerinde fit edilseydi bu esitlik bozulurdu.
    """
    from automl.agents.loader import load
    from automl.agents.planner import plan
    from automl.agents.profiler import profile
    from automl.agents.splitter import split

    st = RunState(data_path="data/iris.csv")
    st = load(st)
    st = profile(st)
    st = split(st)
    st = plan(st)
    st = preprocess(st)

    scaler = _scaler_al(st.preprocessor)
    kolonlar = st.plan.numeric_cols

    train_ort = st.X_train[kolonlar].mean().to_numpy()
    tum_veri_ort = st.df[kolonlar].mean().to_numpy()

    assert np.allclose(scaler.mean_, train_ort)
    # Train ile tum veri ortalamasi ayni olsaydi test bir sey kanitlamazdi
    assert not np.allclose(train_ort, tum_veri_ort)
