"""Gizli sayisal kolon: kirli degerler yuzunden metin okunan sayisal kolonlar."""
import numpy as np
import pandas as pd
import pytest

from automl.agents.profiler import GIZLI_SAYISAL_ESIGI, _detect_type, profile
from automl.schemas import RunState

N = 200


def _sayilar(n: int = N, seed: int = 0) -> list[str]:
    "Ondalikli, yuksek kardinaliteli sayilari string olarak uretir."
    rng = np.random.default_rng(seed)
    return [f"{x:.2f}" for x in rng.uniform(10, 9000, size=n)]


def _kirlet(degerler: list[str], bozuklar: list[str]) -> pd.Series:
    "Ilk len(bozuklar) degeri eksik deger yazimlariyla degistirir."
    return pd.Series(bozuklar + degerler[len(bozuklar):], dtype=object)


def _profil(df: pd.DataFrame, target: str = "hedef"):
    st = RunState(data_path="<test>", target=target)
    st.df = df
    st = profile(st)
    assert st.profile is not None
    return st, {c.name: c for c in st.profile.columns}


def _df(kolon: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"kolon": kolon, "hedef": ["a", "b"] * (N // 2)})


# --- Tip tespiti --------------------------------------------------------

def test_bos_string_iceren_sayisal_kolon_numeric():
    "TotalCharges durumu: birkac ' ' degeri kolonu metne cevirmemeli."
    s = _kirlet(_sayilar(), [" "] * 3)
    assert _detect_type(s, n_rows=N) == "numeric"


def test_na_yazimlari_iceren_sayisal_kolon_numeric():
    "'NA', 'N/A', '-', '?' eksik deger sayilmali, kolon sayisal kalmali."
    s = _kirlet(_sayilar(), ["NA", "N/A", "-", "?", "null", "?"])
    assert _detect_type(s, n_rows=N) == "numeric"


def test_gercek_metin_kolonu_text_kalir():
    "Icinde sayi gecen cumleler sayiya cevrilemez, text kalmali."
    s = pd.Series([f"siparis {i} teslim edildi" for i in range(N)])
    assert _detect_type(s, n_rows=N) == "text"


def test_gercek_kategorik_kolon_categorical_kalir():
    s = pd.Series(["Female", "Male"] * (N // 2))
    assert _detect_type(s, n_rows=N) == "categorical"


def test_string_yazilmis_az_kardinaliteli_sayilar_kategorik_kalir():
    "'1','2','3' sayiya cevrilir ama Kural 2 (oransal esik) kategorik der."
    s = pd.Series(["1", "2", "3"] * 100)
    assert _detect_type(s, n_rows=300) == "categorical"


def test_yarisi_sayi_olan_karisik_kolon_numeric_sanilmaz():
    "%50 cevrilebilen kolon esigin altinda: sayisal sayilmamali."
    sayilar = _sayilar()
    s = pd.Series([sayilar[i] if i % 2 else f"kod_{i}" for i in range(N)])
    assert _detect_type(s, n_rows=N) != "numeric"


def test_esik_siniri_dahil():
    "Dolu degerlerin tam %90'i sayiysa sayisal, %89'u ise degil."
    assert GIZLI_SAYISAL_ESIGI == 0.90
    sayilar = _sayilar(100)
    esitte = _kirlet(sayilar, ["?"] * 10)       # %90
    altinda = _kirlet(sayilar, ["?"] * 11)      # %89
    assert _detect_type(esitte, n_rows=100) == "numeric"
    assert _detect_type(altinda, n_rows=100) != "numeric"


# --- Profil: donusum df'e yansiyor mu? ----------------------------------

def test_profil_kolonu_sayiya_cevirir_ve_isaretler():
    s = _kirlet(_sayilar(), [" ", "?", " ", "inf"])
    st, kolonlar = _profil(_df(s))
    c = kolonlar["kolon"]

    assert c.inferred_type == "numeric"
    assert c.donusturuldu
    assert c.ham_dtype == "object"
    assert c.n_donusmeyen == 4
    assert c.donusum_orani == pytest.approx(196 / 200)
    assert set(c.donusmeyen_ornekler) == {"' '", "'?'", "'inf'"}

    # df gercekten sayisal: sklearn string gormeyecek, bozuklar NaN.
    kolon = st.df["kolon"]
    assert pd.api.types.is_float_dtype(kolon)
    assert int(kolon.isna().sum()) == 4
    assert np.isfinite(kolon.dropna()).all()
    assert c.null_ratio == pytest.approx(4 / 200)
    assert c.mean is not None   # istatistikler cevrilmis degerlerden


def test_kategorik_kalan_sayisal_stringler_donusturulmez():
    "Kategorik kalan kolonun orijinal degerleri korunmali."
    df = pd.DataFrame({"kolon": ["1", "2", "3"] * 100,
                       "hedef": ["a", "b", "c"] * 100})
    st, kolonlar = _profil(df)
    assert kolonlar["kolon"].inferred_type == "categorical"
    assert not kolonlar["kolon"].donusturuldu
    assert list(st.df["kolon"][:3]) == ["1", "2", "3"]


def test_planner_gerekcesi_donusumu_anlatir():
    from automl.agents.planner import plan

    st, _ = _profil(_df(_kirlet(_sayilar(), [" "] * 11)))
    st = plan(st)
    assert st.plan is not None
    assert "kolon" in st.plan.numeric_cols
    karar = next(k for k in st.plan.column_decisions if k.name == "kolon")
    assert karar.decision == "numeric"
    assert "gizli sayısal" in karar.reason
    assert "%94.5" in karar.trigger
    assert "GIZLI_SAYISAL_ESIGI %90" in karar.trigger
    assert "11 değer (' ')" in karar.trigger


# --- Uctan uca: CSV -> profile -> split -> plan -> preprocess -----------

def test_uctan_uca_csvden_gelen_kirli_kolon_sizintisiz_islenir(tmp_path):
    """Kirli kolon CSV'den metin okunur, sayiya cevrilir, pipeline cokmez.

    Imputer'in doldurdugu deger SADECE train'in medyani olmali.
    """
    from automl.agents.loader import load
    from automl.agents.planner import plan
    from automl.agents.preprocessor import preprocess
    from automl.agents.splitter import split

    n = 400
    rng = np.random.default_rng(3)
    degerler = [f"{x:.2f}" for x in rng.lognormal(6, 1, size=n)]
    for i, bozuk in zip(range(0, n, 20), [" ", "-", "?", "NA"] * 5):
        degerler[i] = bozuk
    df = pd.DataFrame({
        "tutar": degerler,
        "diger": rng.normal(size=n),
        "hedef": rng.choice(["evet", "hayir"], size=n),
    })
    yol = tmp_path / "kirli.csv"
    df.to_csv(yol, index=False)
    assert not pd.api.types.is_numeric_dtype(pd.read_csv(yol)["tutar"])

    st = load(RunState(data_path=str(yol), target="hedef"))
    st = split(profile(st))
    st = preprocess(plan(st))
    assert st.plan is not None

    assert "tutar" in st.plan.numeric_cols
    assert pd.api.types.is_float_dtype(st.X_train["tutar"])
    assert not np.isnan(st.X_train_t).any()
    assert not np.isnan(st.X_test_t).any()

    imputer = st.preprocessor.named_transformers_["num"].named_steps["imputer"]
    sira = st.plan.numeric_cols.index("tutar")
    train_medyan = float(st.X_train["tutar"].median())
    tum_medyan = float(st.df["tutar"].median())
    assert imputer.statistics_[sira] == pytest.approx(train_medyan)
    assert train_medyan != pytest.approx(tum_medyan)
