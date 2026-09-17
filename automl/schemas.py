#Bu dosya "veri kutularını tanımlıyor" yani sistemde dolaşacak bilgilerin şekli.

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict

# Kolon ve gorev tipleri tek yerde tanimli: steps.py de bu aliaslari kullanir,
# boylece tip denetleyicisi "str -> Literal" uyusmazligi vermez.
InferredType = Literal["numeric", "categorical", "text", "datetime"]
TaskType = Literal["classification", "regression", "clustering"]
# Bir kolon hakkinda verilebilecek kararlar: atildi, hangi pipeline'a gitti
# ya da hedef kolon oldugu icin ozellik listelerine hic girmedi.
KararTipi = Literal["drop", "numeric", "categorical", "target"]


class _Strict(BaseModel):
    "Yazim hatasi olan alanlar sessizce yutulmasin diye ortak taban."
    model_config = ConfigDict(extra="forbid")

class ColumnProfile(_Strict):
    "Tek bir kolon hakkında öğrendiklerimiz"
    name: str
    dtype: str
    inferred_type: InferredType
    n_unique: int
    null_ratio: float
    # betimsel istatistik (sadece sayisal kolonlarda dolu)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    skew: Optional[float] = None
    # kategorik kolonlar icin
    top_value: Optional[str] = None
    top_ratio: Optional[float] = None
    # Kimlik (sira numarasi) kolonu sinyali: karar profiler'da sadece
    # veriden uretilir, kolon adina bakilmaz.
    is_probable_id: bool = False
    id_ardisik: bool = False   # gerekce metnine girer, karara girmez
    # Gizli sayisal kolon: metin tipinde geldi, sayiya cevrildi. Profil
    # alanlari (dtype, null_ratio, istatistikler) cevrilmis degerlere aittir.
    donusturuldu: bool = False
    ham_dtype: Optional[str] = None           # okunurken gelen dtype
    donusum_orani: Optional[float] = None     # dolu degerlerin sayi orani
    n_donusmeyen: int = 0                     # eksik sayilan deger sayisi
    donusmeyen_ornekler: list[str] = []

class DataProfile(_Strict):
    "Veri setinin tamamı hakkında öğrendiklerimiz"
    n_rows: int
    n_cols: int
    target: Optional[str] = None
    task_type: TaskType
    columns: list[ColumnProfile] = []
    class_balance: Optional[dict[str,float]] = None
    high_correlations: list[tuple[str, str, float]] = []
    # Sinif dengesizligi (sadece classification'da dolu). Karar profiler'da
    # DENGESIZLIK_ORANI * (1/K) esigiyle verilir; metrik secimi buna gore
    # degisir.
    is_imbalanced: bool = False
    imbalance_ratio: Optional[float] = None   # cogunluk / azinlik
    minority_class: Optional[str] = None
    minority_ratio: Optional[float] = None
    imbalance_threshold: Optional[float] = None   # sinif sayisina gore
    imbalance_reason: str = ""


class ColumnDecision(_Strict):
    """Tek bir kolon icin verilen karar ve GEREKCESI.

    Karar mantigi planner'da; bu kayit sadece "ne yapildi, neden yapildi,
    hangi deger tetikledi" uclusunu saklar ki terminale basilabilsin.
    """
    name: str
    inferred_type: InferredType
    decision: KararTipi
    reason: str            # insan okuyabilir gerekce cumlesi
    trigger: str = ""      # tetikleyen deger + karsilastirilan esik


class PreprocessingPlan(BaseModel):
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    drop_cols: list[str] = []
    numeric_imputation: str = "median"
    categorical_imputation: str = "most_frequent"
    scaling: str = "standard"
    encoding: str = "onehot"
    use_pca: bool = False
    n_components: Optional[int] = None
    notes: list[str] = []
    # Karar gerekceleri: mevcut alanlarin yanina eklenir, onlari degistirmez.
    column_decisions: list[ColumnDecision] = []
    pca_reason: str = ""
    step_reasons: dict[str, str] = {}

class ModelScore(BaseModel):
    name: str
    cv_mean: float
    cv_std: float

class ModelSecimi(BaseModel):
    "LLM'in onerdigi aday model listesi."
    models: list[str] = []
    reason: str = ""

class RunResult(BaseModel):
    task_type: str
    candidates: list = field(default_factory=list) 
    model_name: str
    metric_name: str
    metric_value: float
    test_metrics: dict[str, float] = {}
    feature_importance: dict[str, float] = {}
    gini_importance: dict[str, float] = {}
    # Azinlik sinifi confusion matrix sayimlari (sadece dengesiz veride):
    # toplam, yakalanan, kacirilan, yanlis_alarm.
    azinlik_sayimlari: dict[str, int] = {}
    # Secilen modelin CV skoru Baseline'dan ne kadar iyi? (metric_name ile)
    baseline_cv: Optional[float] = None
    baseline_farki: Optional[float] = None
    baseline_uyarisi: str = ""

@dataclass
class RunState:
    "Tüm aşamalar arasında dolaşan çanta"
    data_path: str
    target: Optional[str] = None
    df: Any = None
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None
    preprocessor: Any = None
    X_train_t: Any = None
    X_test_t: Any = None
    model: Any = None
    best_name: Optional[str] = None
    candidates: list = field(default_factory=list)
    sure_sn: float = 0.0
    profile: Optional[DataProfile] = None
    plan: Optional[PreprocessingPlan] = None
    result: Optional[RunResult] = None
    # self-improvement dongusu
    iterasyon: int = 0
    gecmis_denemeler: list = field(default_factory=list)
    strateji: str = "varsayilan"

    
    