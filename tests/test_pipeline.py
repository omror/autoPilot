"""Uctan uca boru hatti: zorlu veriyle cokmeden calisiyor mu?"""
import io
import contextlib
from pathlib import Path

import pytest

from automl.orchestrator import run

MESSY = "data/messy.csv"


@pytest.fixture(scope="module")
def messy_state():
    "messy.csv'yi bir kez uctan uca calistirir."
    if not Path(MESSY).exists():
        pytest.skip("data/messy.csv yok, once make_data.py calistirin")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run(MESSY)


def test_messy_uctan_uca_cokmuyor(messy_state):
    "Eksik deger, metin, tarih ve bos kolon boru hattini cokertmemeli."
    assert messy_state.result is not None
    assert messy_state.plan is not None
    assert messy_state.profile is not None


def test_messy_task_tipi_classification(messy_state):
    "Ikili hedef -> classification olarak taninmali."
    assert messy_state.profile.task_type == "classification"


def test_bos_kolon_atildi(messy_state):
    "%100 null kolon kullanilmamali."
    pl = messy_state.plan
    assert "bos_kolon" in pl.drop_cols
    assert "bos_kolon" not in pl.numeric_cols
    assert "bos_kolon" not in pl.categorical_cols


def test_yuksek_kardinaliteli_kategorik_atildi(messy_state):
    "~60 essiz degerli kategorik kolon one-hot patlamasini onlemek icin atilir."
    assert "urun_kodu" in messy_state.plan.drop_cols


def test_metin_kolonu_atildi(messy_state):
    "Her satirda farkli olan serbest metin kolonu atilmali."
    assert "aciklama" in messy_state.plan.drop_cols


def test_tarih_kolonu_atildi(messy_state):
    "Tarih kolonu ozellik olarak kullanilmamali."
    assert "kayit_tarihi" in messy_state.plan.drop_cols


def test_kullanilabilir_kolonlar_korundu(messy_state):
    "Ise yarar kolonlar atilmamali: 3 sayisal + 1 kategorik kalmali."
    pl = messy_state.plan
    assert set(pl.numeric_cols) == {"olcum_a", "olcum_b", "olcum_c"}
    assert "seviye" in pl.categorical_cols


def test_eksik_degerli_kolon_kullanilmaya_devam_ediyor(messy_state):
    "%15 eksik olan kolon atilmamali, imputation ile kullanilmali."
    assert "olcum_a" in messy_state.plan.numeric_cols


def test_sonuc_uretiliyor(messy_state):
    "Skor uretilmis ve makul araliкta olmali."
    r = messy_state.result
    assert r.metric_name == "f1_weighted"
    assert 0.0 <= r.metric_value <= 1.0
    assert r.model_name


def test_iterasyon_kaydi_tutuluyor(messy_state):
    "Self-improvement dongusu en az bir deneme kaydetmeli."
    denemeler = messy_state.gecmis_denemeler
    assert len(denemeler) >= 1
    assert denemeler[0]["strateji"] == "varsayilan"


def test_donusturulmus_veride_nan_kalmadi(messy_state):
    "Imputation sonrasi egitime giren matriste NaN olmamali."
    import numpy as np

    assert not np.isnan(messy_state.X_train_t).any()
    assert not np.isnan(messy_state.X_test_t).any()
