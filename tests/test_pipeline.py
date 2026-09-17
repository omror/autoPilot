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


def test_yuksek_kardinaliteli_gurultu_kolonu_cv_ile_atilir(messy_state):
    """urun_kodu ~60 kategorili ve hedefle iliskisiz (make_data'da rastgele).

    Nadir toplama stratejisi secilir ama train icindeki CV'de katkisi fold
    oynakliginin altinda kalir: kolon atilir.
    """
    pl = messy_state.plan
    assert "urun_kodu" in pl.drop_cols
    assert pl.nadir_toplama_cols == [] and pl.frekans_cols == []

    karar = next(k for k in pl.column_decisions if k.name == "urun_kodu")
    assert karar.decision == "drop"
    assert "60 kategori > eşik 50" in karar.reason
    assert "CV ile ölçüldü" in karar.reason
    assert "sinyal taşımıyor" in karar.trigger


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


def test_her_kolon_icin_gerekce_kaydi_var(messy_state):
    "Hedef dahil her kolon icin karar + gerekce kaydedilmeli."
    pl, p = messy_state.plan, messy_state.profile
    kararlar = pl.column_decisions
    assert len(kararlar) == p.n_cols
    assert {k.name for k in kararlar} == {c.name for c in p.columns}
    for k in kararlar:
        assert k.reason, f"{k.name}: gerekce bos"
        assert k.trigger, f"{k.name}: tetikleyen deger bos"


def test_drop_gerekceleri_tetiklenen_kurali_gosteriyor(messy_state):
    """Atilan her kolonun gerekcesi hangi kuralin tetiklendigini soylemeli.

    Dikkat: kayit_tarihi datetime olarak parse EDILMIYOR (loader tarih
    cevirmiyor, kolon str kaliyor). 400 essiz string oldugu icin TEXT
    kuraliyla atiliyor. Yani messy.csv'de 3 ayri drop kurali tetiklenir:
    null orani, serbest metin (2 kolon) ve yuksek kardinaliteli kolonun
    CV'de katki saglamamasi.
    """
    atilanlar = {k.name: k for k in messy_state.plan.column_decisions
                 if k.decision == "drop"}
    assert set(atilanlar) == {"bos_kolon", "urun_kodu", "kayit_tarihi",
                              "aciklama"}

    # Tetikleyen deger gercekten esikle birlikte yaziliyor mu?
    assert "%100" in atilanlar["bos_kolon"].trigger
    assert "%50" in atilanlar["bos_kolon"].trigger

    # Metin kuraliyla atilanlar tipini de gerekcesini de metin olarak verir.
    for ad in ("kayit_tarihi", "aciklama"):
        assert atilanlar[ad].inferred_type == "text"
        assert "metin" in atilanlar[ad].reason

    sebepler = {k.reason for k in atilanlar.values()}
    assert len(sebepler) == 3, f"sebepler ayrismamis: {sebepler}"


def test_hedef_kolon_ayri_isaretli(messy_state):
    "Hedef kolon atilmis gibi degil, 'hedef' olarak kaydedilmeli."
    hedef = [k for k in messy_state.plan.column_decisions
             if k.decision == "target"]
    assert len(hedef) == 1
    assert hedef[0].name == "hedef"


def test_pca_ve_adim_gerekceleri_dolu(messy_state):
    "PCA karari ve imputation/scaling/encoding secimleri gerekceli olmali."
    pl = messy_state.plan
    assert pl.pca_reason
    assert set(pl.step_reasons) == {"numeric_imputation",
                                    "categorical_imputation",
                                    "scaling", "encoding"}
    assert pl.step_reasons["numeric_imputation"].startswith(
        pl.numeric_imputation)


def test_donusturulmus_veride_nan_kalmadi(messy_state):
    "Imputation sonrasi egitime giren matriste NaN olmamali."
    import numpy as np

    assert not np.isnan(messy_state.X_train_t).any()
    assert not np.isnan(messy_state.X_test_t).any()
