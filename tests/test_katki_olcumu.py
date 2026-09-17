"""Yuksek kardinaliteli kolonun katkisi: train icinde CV ile olculur."""
import math

import numpy as np
import pandas as pd
import pytest

from automl.agents import planner
from automl.agents.planner import KATKI_KATSAYISI, _KatkiOlcumu, plan
from automl.agents.profiler import profile
from automl.agents.splitter import split
from automl.schemas import RunState


def _nadir_yollu_kategoriler(rng) -> tuple[list[str], np.ndarray]:
    """40 sik (30'ar satir) + 30 nadir (2'ser satir) kategori = 1260 satir.

    Train'de nadirler toplanir, 41 kategori kalir: nadir toplama yolu.
    Donen: (kategoriler, sik kategorinin "evet" egilimi: ilk 20'si evet).
    """
    kat = [f"sik_{i:02d}" for i in range(40) for _ in range(30)]
    kat += [f"nadir_{i:02d}" for i in range(30) for _ in range(2)]
    egilim = np.array([k.startswith("sik_") and int(k[4:]) < 20 for k in kat])
    return kat, egilim


def _frekans_yollu_kategoriler() -> tuple[list[str], np.ndarray]:
    """30 kategori x 24 satir + 30 kategori x 42 satir = 1980 satir.

    Hicbiri nadir degil, 60 kategori kalir: frequency encoding yolu.
    Donen: (kategoriler, kalabalik kategori mi).
    """
    kat = [f"az_{i:02d}" for i in range(30) for _ in range(24)]
    kat += [f"cok_{i:02d}" for i in range(30) for _ in range(42)]
    return kat, np.array([k.startswith("cok_") for k in kat])


def _state(kolon: list[str], hedef: np.ndarray, x: np.ndarray,
           seed: int = 0) -> RunState:
    "Profile + split edilmis state (plan henuz calismadi)."
    df = pd.DataFrame({"kod": kolon, "x": x,
                       "hedef": np.where(hedef, "evet", "hayir")})
    st = RunState(data_path="<test>", target="hedef")
    st.df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return split(profile(st))


def _karar(st: RunState, ad: str):
    assert st.plan is not None
    return next(k for k in st.plan.column_decisions if k.name == ad)


# --- Gurultu atilir, sinyal tutulur -------------------------------------

def test_gurultu_kolonu_atilir():
    "Kategoriler hedefle iliskisiz, hedef sayisal x'ten gelir: kolon atilir."
    rng = np.random.default_rng(0)
    kat, _ = _nadir_yollu_kategoriler(rng)
    x = rng.normal(size=len(kat))
    hedef = x + rng.normal(0, 0.8, size=len(kat)) > 0
    st = plan(_state(kat, hedef, x))

    assert st.plan is not None
    assert "kod" in st.plan.drop_cols
    assert st.plan.nadir_toplama_cols == [] and st.plan.frekans_cols == []
    karar = _karar(st, "kod")
    assert karar.decision == "drop"
    assert "70 kategori > eşik 50" in karar.reason
    assert "nadir toplama + onehot ile kodlanarak" in karar.reason
    assert "sinyal taşımıyor" in karar.trigger
    assert "KATKI_KATSAYISI" in karar.trigger


def test_sinyal_tasiyan_kolon_nadir_toplama_ile_tutulur():
    "Sik kategorilerin yarisi 'evet'e, yarisi 'hayir'a yatkin: kolon kalir."
    rng = np.random.default_rng(1)
    kat, egilim = _nadir_yollu_kategoriler(rng)
    hedef = rng.random(len(kat)) < np.where(egilim, 0.9, 0.1)
    st = plan(_state(kat, hedef, rng.normal(size=len(kat))))

    assert st.plan is not None
    assert st.plan.nadir_toplama_cols == ["kod"]
    karar = _karar(st, "kod")
    assert karar.decision == "nadir_toplama"
    assert "sinyal taşıyor" in karar.reason
    assert "toplama sonrası 41 kategori kaldı" in karar.trigger


def test_sinyal_tasiyan_kolon_frequency_encoding_ile_tutulur():
    "Kalabalik kategoriler 'evet'e yatkin: sinyal frekansta, kolon kalir."
    rng = np.random.default_rng(2)
    kat, kalabalik = _frekans_yollu_kategoriler()
    hedef = rng.random(len(kat)) < np.where(kalabalik, 0.85, 0.15)
    st = plan(_state(kat, hedef, rng.normal(size=len(kat))))

    assert st.plan is not None
    assert st.plan.frekans_cols == ["kod"]
    assert _karar(st, "kod").decision == "frekans"


# --- SIZINTI: olcum sadece train'de ------------------------------------

def test_olcum_sadece_train_setini_gorur(monkeypatch):
    """Iki ayri kontrol:

    1) CV'ye giden her X ve y, train satirlarindan ibaret olmali.
    2) Davranis: train'de hedef sayisal x'ten gelir ve kolon gurultudur
       (messy gibi); test'te ise kategori hedefi BIREBIR belirler (veri yari
       yariya bolunur). Olcum test'i ya da tum veriyi gorseydi kolon sinyalli
       gorunup tutulurdu; sadece train'i gordugu icin atilmali. X_test /
       y_test ayrica gizlenir.
    """
    rng = np.random.default_rng(3)
    kat, egilim = _nadir_yollu_kategoriler(rng)
    n = len(kat)
    df = pd.DataFrame({"kod": kat, "x": rng.normal(size=n)})
    df = df.sample(frac=1, random_state=3).reset_index(drop=True)
    egilim = pd.Series(egilim).sample(frac=1, random_state=3).to_numpy()

    yari = n // 2
    x_train = df["x"].to_numpy()[:yari]
    gurultulu = x_train + rng.normal(0, 0.8, size=yari) > 0  # train: x
    zehirli = egilim[yari:]                            # test: kategori = hedef
    df["hedef"] = np.where(np.concatenate([gurultulu, zehirli]),
                           "evet", "hayir")

    st = RunState(data_path="<test>", target="hedef")
    st.df = df
    st = profile(st)
    X, y = df.drop(columns=["hedef"]), df["hedef"]
    st.X_train, st.y_train = X.iloc[:yari], y.iloc[:yari]
    train_idx, test_idx = X.index[:yari], X.index[yari:]

    gorulen: list[tuple[pd.Index, pd.Index]] = []
    gercek_cv = planner.cross_val_score

    def kaydeden_cv(model, X, y, **kw):
        gorulen.append((X.index, y.index))
        return gercek_cv(model, X, y, **kw)

    monkeypatch.setattr(planner, "cross_val_score", kaydeden_cv)
    st.X_test = None
    st.y_test = None
    st = plan(st)

    assert len(gorulen) == 2       # kolonlu + kolonsuz
    for X_idx, y_idx in gorulen:
        assert X_idx.equals(train_idx) and y_idx.equals(train_idx)
        assert not X_idx.isin(test_idx).any()
    assert _karar(st, "kod").decision == "drop"


# --- Olcume girmeyenler -------------------------------------------------

def _olcum_yasak(monkeypatch):
    def patla(*args, **kwargs):
        raise AssertionError("katki olcumu cagrilmamaliydi")
    monkeypatch.setattr(planner, "_katki_olc", patla)


def test_dusuk_kardinaliteli_kolonlar_olcume_girmez(monkeypatch):
    "Tam esikteki (50 kategori) kolon bile eskisi gibi duz one-hot."
    _olcum_yasak(monkeypatch)
    rng = np.random.default_rng(4)
    kat = [f"k{i:02d}" for i in range(50) for _ in range(20)]
    st = plan(_state(kat, rng.random(len(kat)) < 0.5,
                     rng.normal(size=len(kat))))
    assert st.plan is not None
    assert st.plan.categorical_cols == ["kod"]
    assert _karar(st, "kod").decision == "categorical"


def test_uc_durum_kolonu_cvye_gitmeden_atilir(monkeypatch):
    """Neredeyse her satiri farkli kolon olcumsuz atilir.

    Profiler boyle kolonlari genelde 'text' der; burada kural dogrudan
    sinansin diye tip kategorik olarak isaretlenir.
    """
    _olcum_yasak(monkeypatch)
    rng = np.random.default_rng(5)
    n = 400
    st = _state([f"id_{i}" for i in range(n)], rng.random(n) < 0.5,
                rng.normal(size=n))
    assert st.profile is not None
    st.profile.columns = [
        c.model_copy(update={"inferred_type": "categorical"})
        if c.name == "kod" else c for c in st.profile.columns]
    st = plan(st)

    karar = _karar(st, "kod")
    assert karar.decision == "drop"
    assert "neredeyse her satır farklı" in karar.reason


# --- Anlamli fark kurali ------------------------------------------------

def _olcum(kolonlu, kolonsuz) -> _KatkiOlcumu:
    return _KatkiOlcumu(metrik="f1_weighted", model_kolonlu="LR",
                        model_kolonsuz="LR", kolonlu=kolonlu,
                        kolonsuz=kolonsuz, sure_sn=0.0)


def test_esik_iki_surumun_havuzlanmis_oynakligi():
    o = _olcum((0.80, 0.03), (0.70, 0.04))
    assert o.referans == pytest.approx(math.sqrt((0.03**2 + 0.04**2) / 2))
    assert o.esik == pytest.approx(KATKI_KATSAYISI * o.referans)


def test_telco_mertebesindeki_fark_gurultu_sayilir():
    "Oynaklik +-0.011 iken 0.0001'lik fark anlamli degil."
    assert not _olcum((0.7964, 0.011), (0.7963, 0.011)).anlamli
    assert _olcum((0.8300, 0.011), (0.7963, 0.011)).anlamli


def test_kolon_skoru_dusuruyorsa_anlamli_degil():
    assert not _olcum((0.70, 0.0), (0.80, 0.0)).anlamli
