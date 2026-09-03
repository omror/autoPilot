"""Kolon tipi tespiti: sistemin domain-bagimsizliginin temeli."""
import pandas as pd

from automl.agents.profiler import _detect_type


def test_ondalikli_kolon_numeric():
    "Ondalik deger iceren kolon her zaman sayisal sayilmali."
    s = pd.Series([1.5, 2.3, 3.7, 4.1, 5.9] * 40)
    assert _detect_type(s, n_rows=200) == "numeric"


def test_az_kardinaliteli_integer_categorical():
    "Az essiz degerli tamsayi kolonu kategorik sayilmali (orn. sinif kodu)."
    s = pd.Series([0, 1, 2] * 100)
    # n_rows=300 -> esik = max(2, min(20, 15)) = 15; 3 essiz <= 15
    assert _detect_type(s, n_rows=300) == "categorical"


def test_yuksek_kardinaliteli_integer_numeric():
    "Cok essiz degerli tamsayi kolonu sayisal sayilmali (orn. yas, sayac)."
    s = pd.Series(range(200))
    # n_rows=200 -> esik = max(2, min(20, 10)) = 10; 200 essiz > 10
    assert _detect_type(s, n_rows=200) == "numeric"


def test_string_kolon_categorical():
    "Az essiz degerli metin kolonu kategorik sayilmali."
    s = pd.Series(["dusuk", "orta", "yuksek"] * 50)
    assert _detect_type(s, n_rows=150) == "categorical"


def test_her_satiri_farkli_string_text():
    "Her satirda farkli olan metin kolonu 'text' sayilmali, kategorik degil."
    s = pd.Series([f"serbest metin {i}" for i in range(200)])
    assert _detect_type(s, n_rows=200) == "text"


def test_bool_kolon_categorical():
    "Boolean kolon kategorik sayilmali."
    s = pd.Series([True, False] * 50)
    assert _detect_type(s, n_rows=100) == "categorical"


def test_datetime_kolon_datetime():
    "Gercekten datetime dtype'li kolon datetime sayilmali."
    s = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D"))
    assert _detect_type(s, n_rows=100) == "datetime"


def test_esik_veri_boyutuna_gore_degisir():
    """AYNI kolon, farkli satir sayisinda farkli tip verebilmeli.

    Esik = max(2, min(20, n_rows * 0.05)). Bu yuzden 8 essiz degerli bir
    tamsayi kolonu buyuk veride kategorik, kucuk veride sayisal olur.
    Sabit esik kullanilsaydi bu test gecmezdi.
    """
    s = pd.Series(list(range(8)) * 5)  # 8 essiz tamsayi

    # Buyuk veri: esik = min(20, 50) = 20 -> 8 <= 20 -> kategorik
    assert _detect_type(s, n_rows=1000) == "categorical"

    # Kucuk veri: esik = max(2, min(20, 2)) = 2 -> 8 > 2 -> sayisal
    assert _detect_type(s, n_rows=40) == "numeric"


def test_tamamen_bos_kolon_cokmez():
    "Tamamen bos kolon tip tespitini cokertmemeli."
    s = pd.Series([None] * 100, dtype="float64")
    assert _detect_type(s, n_rows=100) == "numeric"
