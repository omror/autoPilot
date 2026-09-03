"""Gorev tipi ve hedef kolon tespiti."""
import pandas as pd
import pytest

from automl.agents.profiler import profile
from automl.schemas import RunState


def _state(df: pd.DataFrame, target=None) -> RunState:
    st = RunState(data_path="<test>", target=target)
    st.df = df
    return st


def test_sayisal_hedef_regression():
    "Surekli sayisal hedef -> regression."
    df = pd.DataFrame({
        "x1": range(200),
        "x2": [i * 0.5 for i in range(200)],
        "fiyat": [i * 1.7 + 3.2 for i in range(200)],
    })
    st = profile(_state(df, target="fiyat"))
    assert st.profile is not None
    assert st.profile.task_type == "regression"


def test_kategorik_hedef_classification():
    "Az essiz degerli hedef -> classification."
    df = pd.DataFrame({
        "x1": range(200),
        "x2": [i * 0.5 for i in range(200)],
        "sinif": ["a", "b"] * 100,
    })
    st = profile(_state(df, target="sinif"))
    assert st.profile is not None
    assert st.profile.task_type == "classification"
    assert st.profile.class_balance is not None


def test_hedef_belirtilmezse_son_kolon():
    "Hedef verilmezse son kolon hedef kabul edilmeli."
    df = pd.DataFrame({
        "a": range(100),
        "b": range(100),
        "son_kolon": ["x", "y"] * 50,
    })
    st = profile(_state(df, target=None))
    assert st.profile is not None
    assert st.profile.target == "son_kolon"


def test_olmayan_hedef_anlamli_hata():
    "Veride olmayan hedef verilirse acik bir hata firlatmali."
    df = pd.DataFrame({"a": range(50), "b": range(50)})
    with pytest.raises(ValueError) as e:
        profile(_state(df, target="olmayan_kolon"))

    mesaj = str(e.value)
    assert "olmayan_kolon" in mesaj      # hangi kolon eksik
    assert "Mevcut kolonlar" in mesaj    # ne varmis
    assert "a" in mesaj and "b" in mesaj


def test_profile_calismadan_split_hata():
    "Profile uretilmeden plan asamasi calisamaz."
    from automl.agents.planner import plan

    with pytest.raises(RuntimeError):
        plan(RunState(data_path="<test>"))
