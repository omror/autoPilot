"""Veriyi tanir: kolon tipleri, hedef kolon ve gorev tipi."""
import numpy as np
import pandas as pd
from automl.agents.base import Agent
from automl.schemas import (
    ColumnProfile,
    DataProfile,
    InferredType,
    RunState,
    TaskType,
)


def _detect_type(s: pd.Series, n_rows:int) -> InferredType:
    "Bir kolonun gerçek tipini otomatik olarak tespit eder."

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        n_unique = s.nunique(dropna=True)
        if n_unique > 50 and n_unique / max(n_rows, 1) > 0.5:
            return "text"
        return "categorical"

    if pd.api.types.is_bool_dtype(s):
        return "categorical"

    if pd.api.types.is_numeric_dtype(s):
        clean = s.dropna()
        if len(clean) == 0:
            return "numeric"

        # Kural 1: Ondalıklı değer varsa kesin sayısal
        if not clean.mod(1).eq(0).all():
            return "numeric"

        #Kural 2: eşik veri boyutuna göre otomatik hesaplanır.
        esik = max(2, min(20, int(n_rows * 0.05)))
        return "categorical" if clean.nunique() <= esik else "numeric"

    return "categorical"

# Kimlik kolonu tespiti. Esikler ORANSAL: sabit sayi kullanilmaz ki
# veri boyutundan bagimsiz calissin.
ID_ORAN_ESIGI = 0.95   # essiz deger / satir sayisi bu oranin ustundeyse
ID_MIN_SATIR = 20      # bu satirdan az veride ID tespiti hic yapilmaz
ID_ARDISIK_TOLERANS = 0.05   # (max-min+1) satir sayisindan bu kadar sapabilir


def _probable_id(s: pd.Series, n_rows: int,
                 tip: InferredType) -> tuple[bool, bool]:
    """Kolon bir kimlik (sira numarasi) kolonu mu?

    Karar SADECE veriden gelir; kolon adina BAKILMAZ, cunku "id" kelimesi
    aramak domain-bagimsizlik ilkesini bozar.

    Donen: (id_mi, ardisik_mi). Ardisiklik karara GIRMEZ, sadece gerekce
    metninde tetikleyen deger olarak gosterilir.

    Kucuk veride her sayisal kolon "benzersiz" gorunur: 6 satirlik veride
    yas ve maas kolonlari da ID sanilirdi. ID_MIN_SATIR bunu engeller.
    """
    if tip != "numeric" or n_rows < ID_MIN_SATIR:
        return False, False

    # Kimlik kolonunda eksik deger olmaz.
    if bool(s.isna().any()):
        return False, False

    clean = s.dropna()
    if len(clean) == 0:
        return False, False

    # Ondalikli deger varsa olcumdur, sira numarasi degil.
    if not bool(clean.mod(1).eq(0).all()):
        return False, False

    if clean.nunique() / max(n_rows, 1) < ID_ORAN_ESIGI:
        return False, False

    # Ek sinyal: degerler ardisik mi (1..n gibi)? Sadece bilgi amacli.
    aralik = float(clean.max()) - float(clean.min()) + 1
    ardisik = abs(aralik - n_rows) / max(n_rows, 1) < ID_ARDISIK_TOLERANS
    return True, ardisik


def _numeric_stats(s: pd.Series) -> dict:
    """Sayisal kolon icin betimsel istatistikler."""
    clean = s.dropna()
    if len(clean) == 0:
        return {}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,  # type: ignore
        "min": float(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "skew": float(clean.skew()) if len(clean) > 2 else 0.0,  # type: ignore
    }


def _categorical_stats(s: pd.Series) -> dict:
    """Kategorik kolon icin en sik deger ve orani."""
    clean = s.dropna()
    if len(clean) == 0:
        return {}
    vc = clean.value_counts(normalize=True)
    return {
        "top_value": str(vc.index[0]),
        "top_ratio": float(vc.iloc[0]),
    }


def _high_correlations(df: pd.DataFrame, cols: list[str],
                       esik: float = 0.85) -> list[tuple[str, str, float]]:
    """Yuksek korelasyonlu sayisal kolon ciftlerini bulur."""
    if len(cols) < 2:
        return []
    try:
        corr = df[cols].corr()  # type: ignore
    except Exception:
        return []

    ciftler = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            deger = corr.loc[a, b]
            if pd.notna(deger) and abs(float(deger)) >= esik:
                ciftler.append((str(a), str(b), round(float(deger), 3)))

    ciftler.sort(key=lambda x: abs(x[2]), reverse=True)
    return ciftler[:15]


# Sinif dengesizligi: en kucuk sinifin orani, dengeli dagilimda bekledigi
# payin (1/K) bu kesrinin ALTINDAYSA veri dengesiz sayilir:
#     esik = DENGESIZLIK_ORANI * (1 / K)
# Esik sinif sayisiyla olceklenir; mutlak bir esik cok sinifli dengeli
# veride yanilirdi (8 dengeli sinifin her biri %12.5 pay alir).
# Oranin gerekcesi ikili veriden gelir (K=2 -> esik 0.15): hep cogunluk
# sinifini soyleyen model azinlik orani p iken f1_weighted =
# (1-p) * 2(1-p)/(2-p) alir: p=0.15'te 0.78, p=0.137'de 0.80
# (classification iterasyon esigi), p=0.10'da 0.85. Yaygin %10 esigi,
# hicbir sey ogrenmeyen modelin esigi gectigi %10-%13.7 araligini
# kacirirdi; %15 bu boslugu kapatir.
DENGESIZLIK_ORANI = 0.30


def dengesizlik_esigi(k: int) -> float:
    "K sinifli veride azinlik sinifi icin dengesizlik esigi."
    return DENGESIZLIK_ORANI * (1 / k)


def _yuzde(x: float) -> str:
    "Esik gibi veriye gore degisen oranlari kisa yuzde olarak yazar."
    return f"%{x * 100:.3g}"


def _cogunluk_modeli_skorlari(oranlar: dict[str, float]) -> tuple[float, float]:
    """Hep cogunluk sinifini soyleyen modelin (f1_weighted, f1_macro) skoru.

    Bu model sadece cogunluk sinifinda isabet eder: o sinifta
    precision = p_max, recall = 1; diger siniflarin F1'i 0. Model
    egitmeden, sadece sinif dagilimindan hesaplanir.
    """
    p_max = max(oranlar.values())
    f1_cogunluk = 2 * p_max / (1 + p_max)
    return p_max * f1_cogunluk, f1_cogunluk / len(oranlar)


def esik_gerekcesi(k: int) -> str:
    """Esik neden bu deger? Metin sabitin kendisinden uretilir.

    k: verideki sinif sayisi; hesaplanan esik metinde gorunur.
    """
    ikili = dengesizlik_esigi(2)
    w, m = _cogunluk_modeli_skorlari({"cogunluk": 1 - ikili,
                                      "azinlik": ikili})
    return (
        f"Eşik sınıf sayısıyla ölçeklenir: eşik = DENGESIZLIK_ORANI "
        f"({_yuzde(DENGESIZLIK_ORANI)}) × 1/K. Dengeli dağılımda her sınıf "
        f"1/K pay alır; sabit bir eşik çok sınıflı dengeli veriyi yanlışlıkla "
        f"dengesiz sayardı. Bu veride K={k}: beklenen pay "
        f"{_yuzde(1 / k)}, eşik {_yuzde(dengesizlik_esigi(k))}. Oran ikili "
        f"veriden gelir: K=2'de eşik {_yuzde(ikili)} ve azınlık tam bu "
        f"orandayken bile hep çoğunluk sınıfını söyleyen model "
        f"f1_weighted={w:.2f} alır (f1_macro={m:.2f}). Azınlık küçüldükçe "
        f"bu skor 1'e yaklaşır; eşiğin altında f1_weighted azınlık sınıfını "
        f"fiilen göremez."
    )


def _dengesizlik(oranlar: dict[str, float]) -> dict:
    """Sinif dagilimindan dengesizlik alanlarini uretir.

    Cok sinifli veride de ayni kural gecerli: en kucuk sinifin oranina
    bakilir, esik sinif sayisina gore olceklenir.
    """
    if not oranlar:
        return {}
    azinlik = min(oranlar, key=lambda k: oranlar[k])
    cogunluk = max(oranlar, key=lambda k: oranlar[k])
    p_min, p_max = oranlar[azinlik], oranlar[cogunluk]
    k = len(oranlar)
    esik_degeri = dengesizlik_esigi(k)
    dengesiz = p_min < esik_degeri
    esik = (f"(K={k} sınıf, eşik = DENGESIZLIK_ORANI "
            f"{_yuzde(DENGESIZLIK_ORANI)} × 1/{k} = {_yuzde(esik_degeri)})")

    if dengesiz:
        # Azinlik sinifinin F1'i 0, digerleri en iyi ihtimalle 1 olsa:
        # f1_weighted <= 1 - p_min, f1_macro <= (K-1)/K. Her K icin gecerli.
        gerekce = (
            f"En küçük sınıf '{azinlik}': oran %{p_min * 100:.2f}, eşiğin "
            f"altında {esik}. f1_weighted her sınıfı payı kadar "
            f"ağırlıklandırır: '{azinlik}' skora %{p_min * 100:.2f} ağırlıkla "
            f"girer, f1_macro'da ise 1/K = {_yuzde(1 / k)} ağırlıkla. Bu "
            f"sınıfı tamamen kaçıran bir model f1_weighted ile yine de "
            f"{1 - p_min:.4f} kadar yüksek skor alabilir; f1_macro ile en "
            f"fazla {(k - 1) / k:.4f} alır, kaçırılan sınıf skora yansır. Bu "
            f"yüzden CV, test ve iterasyon eşiği f1_weighted yerine f1_macro "
            f"ile değerlendirilir."
        )
    else:
        gerekce = (
            f"En küçük sınıf '{azinlik}': oran %{p_min * 100:.1f}, eşiğin "
            f"altında değil {esik}. Dengeli sayıldı, f1_weighted korunuyor."
        )

    return {
        "is_imbalanced": dengesiz,
        "imbalance_ratio": p_max / p_min,
        "minority_class": azinlik,
        "minority_ratio": p_min,
        "imbalance_threshold": esik_degeri,
        "imbalance_reason": gerekce,
    }


def ana_metrik(p: DataProfile) -> str:
    """CV skoru, test ana metrigi ve iterasyon esigi icin TEK metrik karari.

    Adlar sklearn scoring adlariyla aynidir; modeler dogrudan
    cross_val_score'a verir, evaluator test_metrics anahtari olarak kullanir.
    """
    if p.task_type == "regression":
        return "r2"
    return "f1_macro" if p.is_imbalanced else "f1_weighted"

def profile(state: RunState) -> RunState:

    df = state.df
    n_rows = len(df)

    columns = []
    for name in df.columns:
        s = df[name]
        tip = _detect_type(s, n_rows)
        id_mi, id_ardisik = _probable_id(s, n_rows, tip)

        # Tipe gore ek istatistikler
        ekstra = {}
        if tip == "numeric":
            ekstra = _numeric_stats(s)
        elif tip == "categorical":
            ekstra = _categorical_stats(s)

        columns.append(
            ColumnProfile(
                name=name,
                dtype=str(s.dtype),
                inferred_type=tip,
                n_unique=int(s.nunique(dropna=True)),
                null_ratio=float(s.isna().mean()),
                is_probable_id=id_mi,
                id_ardisik=id_ardisik,
                **ekstra,
            )
        )

    target = state.target
    if target is None and len(df.columns) > 0:
        target = df.columns[-1]

    class_balance = None
    dengesizlik = {}
    task_type: TaskType
    if target is None:
        task_type = "clustering"
    else:
        hedef_profil = next((c for c in columns if c.name == target), None)
        if hedef_profil is None:
            raise ValueError(
                f"Hedef kolon '{target}' veri setinde yok. "
                f"Mevcut kolonlar: {list(df.columns)}"
            )
        if hedef_profil.inferred_type == "numeric":
            task_type = "regression"
        else:
            task_type = "classification"
            oranlar = df[target].value_counts(normalize=True)
            class_balance = {str(k): float(v) for k, v in oranlar.items()}
            dengesizlik = _dengesizlik(class_balance)

        # Sayisal kolonlar arasi yuksek korelasyonlar (hedef haric)
    # Kimlik kolonu korelasyona girmez: sira numarasinin korelasyonu gurultu.
    sayisal = [c.name for c in columns
               if c.inferred_type == "numeric" and not c.is_probable_id
               and c.name != target]
    korelasyonlar = _high_correlations(df, sayisal)

    state.profile = DataProfile(
        n_rows = n_rows,
        n_cols = len(df.columns),
        target = target,
        task_type = task_type,
        columns = columns,
        class_balance=class_balance,
        high_correlations=korelasyonlar,
        **dengesizlik,
    )

    return state


class ProfilerAgent(Agent):
    """Veriyi profiller: kolon tipleri, hedef, gorev tipi."""

    name = "profiler"

    def run(self, state: RunState) -> RunState:
        return profile(state)
