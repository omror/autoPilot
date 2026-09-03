"Gerçek boyutlu test veri setleri üretir."
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_classification,
)

def kaydet(loader, isim, hedef_adi):
    d = loader()
    df = pd.DataFrame(d.data, columns=d.feature_names)
    df[hedef_adi] = d.target
    yol = f"data/{isim}.csv"
    df.to_csv(yol, index = False)
    print(f"{yol:30} {df.shape[0]:4} satir, {df.shape[1]:3} kolon")


kaydet(load_iris, "iris", "tur")
kaydet(load_breast_cancer, "cancer", "teshis")
kaydet(load_diabetes, "diabetes", "ilerleme")


def kaydet_nonlinear():
    "Agac modellerinin kazanmasi beklenen dogrusal olmayan veri seti."
    X, y = make_classification(
        n_samples=800,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=3,
        class_sep=0.7,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["hedef"] = y
    yol = "data/nonlinear.csv"
    df.to_csv(yol, index=False)
    print(f"{yol:30} {df.shape[0]:4} satir, {df.shape[1]:3} kolon")


kaydet_nonlinear()


def kaydet_messy():
    """Eksik deger ve karisik tip iceren zorlu veri seti.

    Planner'in drop kararlarini test eder: bos kolon, yuksek
    kardinaliteli kategorik, metin ve tarih kolonlari atilmali.
    """
    rng = np.random.default_rng(42)
    n = 400

    # 3 sayisal kolon, biri %15 eksik
    sayisal1 = rng.normal(50, 12, n)
    sayisal2 = rng.exponential(3, n)
    sayisal3 = rng.normal(0, 1, n)
    eksik_idx = rng.choice(n, size=int(n * 0.15), replace=False)
    sayisal1[eksik_idx] = np.nan

    # 2 kategorik: biri %8 eksik, biri yuksek kardinaliteli (~60 essiz)
    kategorik_az = rng.choice(["dusuk", "orta", "yuksek"], n).astype(object)
    eksik_kat = rng.choice(n, size=int(n * 0.08), replace=False)
    kategorik_az[eksik_kat] = None
    kategorik_cok = rng.choice([f"kod_{i:02d}" for i in range(60)], n)

    # Hedef: ikili siniflandirma, sayisal kolonlarla iliskili olsun
    sinyal = (np.nan_to_num(sayisal1, nan=50.0) - 50) / 12 + sayisal3
    hedef = (sinyal + rng.normal(0, 0.8, n) > 0).astype(int)

    df = pd.DataFrame({
        "olcum_a": sayisal1,
        "olcum_b": sayisal2,
        "olcum_c": sayisal3,
        "seviye": kategorik_az,
        "urun_kodu": kategorik_cok,
        "kayit_tarihi": pd.date_range("2024-01-01", periods=n, freq="D"),
        "aciklama": [f"serbest metin kaydi numarasi {i}" for i in range(n)],
        "bos_kolon": [None] * n,
        "hedef": hedef,
    })

    yol = "data/messy.csv"
    df.to_csv(yol, index=False)
    print(f"{yol:30} {df.shape[0]:4} satir, {df.shape[1]:3} kolon")


kaydet_messy()
