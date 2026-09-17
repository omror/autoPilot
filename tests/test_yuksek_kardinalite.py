"""Yuksek kardinaliteli kategorikler: nadir toplama, frequency encoding, atma."""
import numpy as np
import pandas as pd
import pytest

from automl.agents import planner
from automl.agents.planner import (KARDINALITE_ESIGI, NADIR_KATEGORI_ESIGI,
                                   TEKIL_SATIR_ESIGI,
                                   _yuksek_kardinalite_karari, plan)
from automl.agents.preprocessor import FrekansKodlayici, preprocess
from automl.agents.profiler import profile
from automl.agents.splitter import split
from automl.schemas import ColumnProfile, RunState


@pytest.fixture(autouse=True)
def katki_anlamli(monkeypatch):
    """Bu dosya strateji MEKANIGINI test eder (secim, encoder, sizinti).

    Kolonlar rastgele hedefle uretildigi icin CV katki olcumu onlari
    (dogru olarak) atardi. Burada olcum "anlamli" sabitlenir; olcumun
    kendisi tests/test_katki_olcumu.py'de test edilir.
    """
    def sabit_olcum(kolon, strateji, taban, X_train, y_train, p):
        return planner._KatkiOlcumu(
            metrik="f1_weighted", model_kolonlu="LogisticRegression",
            model_kolonsuz="LogisticRegression", kolonlu=(0.9, 0.0),
            kolonsuz=(0.5, 0.0), sure_sn=0.0)

    monkeypatch.setattr(planner, "_katki_olc", sabit_olcum)


def _nadirli_kolon() -> list[str]:
    "20 sik kategori (45'er satir) + 50 nadir kategori (2'ser satir) = 1000."
    degerler = [f"sik_{i:02d}" for i in range(20) for _ in range(45)]
    degerler += [f"nadir_{i:02d}" for i in range(50) for _ in range(2)]
    return degerler


def _duz_dagilimli_kolon() -> list[str]:
    "60 kategori x 14-19 satir = 1000; hicbiri nadir degil, toplama yetmez."
    sayimlar = [14 + i % 5 + (1 if i < 40 else 0) for i in range(60)]
    assert sum(sayimlar) == 1000
    return [f"kod_{i:02d}" for i, n in enumerate(sayimlar) for _ in range(n)]


def _hazirla(kolonlar: dict[str, list], seed: int = 0) -> RunState:
    "Profile + split + plan + preprocess edilmis state."
    n = len(next(iter(kolonlar.values())))
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(kolonlar)
    df["x"] = rng.normal(size=n)
    df["hedef"] = rng.choice(["evet", "hayir"], size=n)
    st = RunState(data_path="<test>", target="hedef")
    st.df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return preprocess(plan(split(profile(st))))


def _karar(st: RunState, ad: str):
    assert st.plan is not None
    return next(k for k in st.plan.column_decisions if k.name == ad)


def _adim(st: RunState, dal: str, adim: str):
    return st.preprocessor.named_transformers_[dal].named_steps[adim]


# --- Karar sirasi -------------------------------------------------------

def test_nadir_kategorili_kolon_toplanip_onehota_gider():
    st = _hazirla({"urun": _nadirli_kolon()})
    assert st.plan is not None
    assert st.plan.nadir_toplama_cols == ["urun"]
    assert "urun" not in st.plan.drop_cols + st.plan.categorical_cols

    # Sayimlar train'den: 2 satirlik nadir kategorilerin bazilari tamamen
    # test'e dusmus olabilir, gerekce bunu gercek train sayilariyla yazar.
    sayim = st.X_train["urun"].value_counts()
    nadir = int((sayim < NADIR_KATEGORI_ESIGI * len(st.X_train)).sum())
    assert nadir >= 40 and len(sayim) - nadir == 20

    karar = _karar(st, "urun")
    assert karar.decision == "nadir_toplama"
    assert "tüm veride 70 kategori > eşik 50" in karar.reason
    assert (f"train'deki {len(sayim)} kategoriden {nadir} tanesi %1 "
            f"(NADIR_KATEGORI_ESIGI) altında") in karar.reason
    assert "toplama sonrası 21 kategori kaldı" in karar.trigger


def test_nadir_kategoriler_gercekten_tek_kategoride_toplanir():
    "Train'deki nadir kategoriler birlesir: 20 sik + 1 nadir = 21 kolon."
    st = _hazirla({"urun": _nadirli_kolon()})
    encoder = _adim(st, "nadir", "encoder")
    assert encoder.min_frequency == NADIR_KATEGORI_ESIGI
    train_nadirleri = {k for k in st.X_train["urun"].unique()
                       if k.startswith("nadir_")}
    assert set(encoder.infrequent_categories_[0]) == train_nadirleri
    # urun one-hot'i (21) + x sayisal kolonu (1)
    assert st.X_train_t.shape[1] == 22


def test_toplama_yetmeyen_kolon_frequency_encodinge_gider():
    st = _hazirla({"kod": _duz_dagilimli_kolon()})
    assert st.plan is not None
    assert st.plan.frekans_cols == ["kod"]
    karar = _karar(st, "kod")
    assert karar.decision == "frekans"
    assert f"> eşik {KARDINALITE_ESIGI}" in karar.trigger
    # 60 kategori tek sayisal kolona iner: kod (1) + x (1)
    assert st.X_train_t.shape[1] == 2


def test_neredeyse_her_satiri_farkli_kolon_atilir():
    "Train satirlarinin %90'indan fazlasi tekil degerse strateji denenmez."
    n = 400
    X_train = pd.DataFrame({"kimlik": [f"k{i}" for i in range(n)]})
    c = ColumnProfile(name="kimlik", dtype="object",
                      inferred_type="categorical", n_unique=n,
                      null_ratio=0.0)
    karar, sebep, tetik = _yuksek_kardinalite_karari(c, X_train)
    assert karar == "drop"
    assert "neredeyse her satır farklı" in sebep
    assert (f"%100 > TEKIL_SATIR_ESIGI %{TEKIL_SATIR_ESIGI * 100:g}"
            in tetik)


def test_dusuk_kardinaliteli_kolonlar_etkilenmez():
    "Esigin altindaki kolon eskisi gibi duz one-hot: nadir toplama yok."
    rng = np.random.default_rng(1)
    st = _hazirla({"renk": list(rng.choice(["kirmizi", "mavi", "yesil"],
                                           size=500))})
    pl = st.plan
    assert pl is not None
    assert pl.categorical_cols == ["renk"]
    assert pl.nadir_toplama_cols == [] and pl.frekans_cols == []
    assert _karar(st, "renk").decision == "categorical"
    assert set(st.preprocessor.named_transformers_) == {"num", "cat"}
    encoder = _adim(st, "cat", "encoder")
    assert encoder.min_frequency is None
    assert encoder.handle_unknown == "ignore"


def test_feature_adlari_uretilir():
    "Evaluator'daki importance hesabi feature adlarina dayanir."
    st = _hazirla({"urun": _nadirli_kolon(), "kod": _duz_dagilimli_kolon()})
    adlar = list(st.preprocessor.get_feature_names_out())
    assert len(adlar) == st.X_train_t.shape[1]
    assert any(a.startswith("nadir__urun_") for a in adlar)
    assert "nadir__urun_infrequent_sklearn" in adlar


# --- Test setinde gorulmeyen kategori ------------------------------------

def test_testte_gorulmeyen_kategori_cokmez():
    st = _hazirla({"urun": _nadirli_kolon(), "kod": _duz_dagilimli_kolon()})
    assert st.plan is not None
    test = st.X_test.copy()
    test["urun"] = "hic_gorulmemis"
    test["kod"] = "bu_da_yok"

    cikti = st.preprocessor.transform(test)
    assert not np.isnan(cikti).any()

    # Nadir dal: gorulmeyen kategori nadir grubuna duser.
    adlar = list(st.preprocessor.get_feature_names_out())
    nadir_sutun = adlar.index("nadir__urun_infrequent_sklearn")
    assert (cikti[:, nadir_sutun] == 1).all()

    # Frekans dal: gorulmeyen kategori 0 frekans alir.
    kodlayici = _adim(st, "frekans", "encoder")
    assert kodlayici.transform(np.array([["bu_da_yok"]]))[0, 0] == 0.0


# --- SIZINTI: her sey sadece train'den ogrenilir -------------------------

def test_frekanslar_sadece_trainden_ogrenilir():
    "Test setindeki dagilim, train'den ogrenilen frekansi degistirmemeli."
    train = np.array([["a"]] * 1 + [["b"]] * 9, dtype=object)   # a: %10
    test = np.array([["a"]] * 10, dtype=object)                  # a: %100
    kodlayici = FrekansKodlayici().fit(train)
    assert kodlayici.transform(test)[:, 0] == pytest.approx([0.1] * 10)


def test_nadir_listesi_sadece_trainden_ogrenilir():
    "Train'de nadir olan kategori test'te baskin olsa da nadir kalmali."
    from automl.agents.preprocessor import _make_nadir_onehot

    train = np.array([[f"k{i % 20}"] for i in range(1000)] + [["x"]],
                     dtype=object)                                # x: 1/1001
    encoder = _make_nadir_onehot().fit(train)
    assert "x" in encoder.infrequent_categories_[0]

    test = np.array([["x"]] * 50, dtype=object)
    adlar = list(encoder.get_feature_names_out())
    sutun = adlar.index("x0_infrequent_sklearn")
    assert (encoder.transform(test)[:, sutun] == 1).all()


def test_uctan_uca_frekanslar_train_dagilimina_esit_tum_veriye_degil():
    st = _hazirla({"kod": _duz_dagilimli_kolon()}, seed=3)
    kodlayici = _adim(st, "frekans", "encoder")
    ogrenilen = pd.Series(kodlayici.frekanslar_[0])

    train_oran = st.X_train["kod"].value_counts(normalize=True)
    tum_oran = st.df["kod"].value_counts(normalize=True)
    pd.testing.assert_series_equal(ogrenilen.sort_index(),
                                   train_oran.sort_index(),
                                   check_names=False)
    # Kurgu anlamli olsun: train ve tum veri oranlari gercekten farkli.
    assert not np.allclose(train_oran.sort_index(),
                           tum_oran.reindex(train_oran.index).sort_index())
