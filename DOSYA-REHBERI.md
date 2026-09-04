# Dosya Rehberi

Projedeki her dosyanin ne yaptigini gosteren hizli referans.
Tum aciklamalar dosyalarin **gercek icerigi** okunarak yazildi.

---

## Bolum 1: Genel Bakis

```
autoPilot/
├── automl/                    ← Ana paket: tum sistem mantigi burada
│   ├── __init__.py            (bos — sadece paket isareti)
│   ├── schemas.py             Veri sozlesmeleri (tum dosyalarin ortak dili)
│   ├── orchestrator.py        Ajanlari sirayla calistirir + iterasyon dongusu
│   ├── llm.py                 LLM cagrisi icin tek arayuz (opsiyonel katman)
│   ├── agents/                ← 8 boru hatti adimi
│   │   ├── __init__.py        Toplu import (KULLANILMIYOR — Bolum 9'a bak)
│   │   ├── base.py            Agent soyut sinifi
│   │   ├── loader.py          1. CSV oku
│   │   ├── profiler.py        2. Tip + task tespiti, istatistik, korelasyon
│   │   ├── splitter.py        3. Train/test ayrimi
│   │   ├── planner.py         4. Preprocessing plani
│   │   ├── preprocessor.py    5. Plani sklearn Pipeline'a cevir
│   │   ├── modeler.py         6. Model karsilastirma ve egitim
│   │   ├── evaluator.py       7. Test skorlari + feature importance
│   │   └── recorder.py        8. Diske kaydet
│   └── memory/                ← Kalici hafiza
│       ├── __init__.py        (bos)
│       ├── logger.py          Run'i diske yazar
│       └── store.py           Gecmis run'lari okur, benzerlerini bulur
├── tests/                     ← 30 test
│   ├── conftest.py            Ortak fixture (LLM'i kapatir)
│   ├── test_type_detection.py 9 test — tip tespiti
│   ├── test_task_detection.py 5 test — hedef ve gorev tespiti
│   ├── test_no_leakage.py     5 test — VERI SIZINTISI (en kritik)
│   └── test_pipeline.py       11 test — uctan uca zorlu veri
├── data/                      ← 7 CSV veri seti (git'te degil)
├── runs/                      ← 50 gecmis run kaydi (git'te degil)
├── make_data.py               Test veri setlerini uretir
├── report.py                  Tum veri setlerini calistirip REPORT.md uretir
├── show_runs.py               Gecmis run'lari listeler
├── requirements.txt           Bagimliliklar (surumleriyle)
├── README.md                  Kullanim kilavuzu
├── REPORT.md                  Karsilastirmali sonuc tablosu (uretilen dosya)
├── SUNUM.md                   Sunum ve savunma dokumani
└── .gitignore                 Git disi birakilanlar
```

| Klasor | Tek cumlelik amaci |
|---|---|
| `automl/` | Sistemin tum mantigi; disaridan `python3 -m automl.orchestrator` ile calistirilir |
| `automl/agents/` | Boru hattinin 8 adimi; her biri `RunState` alip `RunState` dondurur |
| `automl/memory/` | Run'lari diske yazma ve gecmisten benzer run bulma |
| `tests/` | Kritik yollarin otomatik korumasi (tip tespiti, task tespiti, sizinti, uctan uca) |
| `data/` | Girdi CSV dosyalari — `make_data.py` uretir, `.gitignore`'da |
| `runs/` | Her calistirmanin kalici kaydi — `.gitignore`'da |

> **Not:** Istediginiz listede `README.tr.md` vardi ama **boyle bir dosya yok**.
> Projede tek README var (`README.md`) ve zaten Turkce yazilmis.

---

## Bolum 2: Calisma Akisi (EN ONEMLI BOLUM)

`python3 -m automl.orchestrator --data data/iris.csv` komutu
calistiginda adim adim ne oluyor:

### Adim 0 — Giris noktasi

`orchestrator.py:238-244` (`if __name__ == "__main__"`)

```python
parser.add_argument("--data", required=True)
parser.add_argument("--target", default=None)
parser.add_argument("--max-iterasyon", type=int, default=3)
run(args.data, args.target, args.max_iterasyon)
```

→ `run("data/iris.csv", None, 3)` cagrilir.

### Adim 1 — `RunState` olusturulur

`orchestrator.py:56`

```python
state = RunState(data_path="data/iris.csv", target=None)
```

`schemas.py:80-103`'teki dataclass. Su an sadece `data_path` dolu;
diger 20 alan bos/varsayilan. Bu nesne bundan sonra **tum ajanlar
arasinda dolasacak**.

### Adim 2 — HAZIRLIK grubu (bir kez calisir)

`orchestrator.py:61-63`

```python
for agent in HAZIRLIK:          # [LoaderAgent, ProfilerAgent, SplitterAgent]
    state = agent(state)
```

`agent(state)` cagrisi `base.py:15-16`'daki `__call__`'a gider, o da
`run(state)`'i cagirir.

**2a. `LoaderAgent` → `loader.load()`** (`loader.py:7-9`)
- Girdi: `state.data_path`
- Yapar: `pd.read_csv("data/iris.csv")`
- Cikti: `state.df` doldu (150 satir x 5 kolon)

**2b. `ProfilerAgent` → `profiler.profile()`** (`profiler.py:93-158`)
- Her kolon icin `_detect_type(s, n_rows)` cagrilir (`profiler.py:14-42`).
  iris'te 4 kolon ondalikli → `numeric`; `tur` kolonu 0/1/2 tamsayi,
  esik `max(2, min(20, 150*0.05)) = 7`, 3 essiz ≤ 7 → `categorical`.
- Tipe gore `_numeric_stats` (`:44-59`) veya `_categorical_stats`
  (`:62-72`) cagrilir.
- `target` verilmedigi icin **son kolon** secilir → `tur`
  (`profiler.py:121-123`).
- `tur` kategorik oldugu icin `task_type = "classification"`
  (`:136-141`), sinif dagilimi hesaplanir.
- `_high_correlations` (`:75-91`) sayisal kolonlar arasi |r| >= 0.85
  ciftleri bulur.
- Cikti: `state.profile` doldu (`DataProfile`)

**2c. `SplitterAgent` → `splitter.split()`** (`splitter.py:7-38`)
- `X = df.drop(columns=["tur"])`, `y = df["tur"]`
- Siniflandirma ve en az kalabalik sinifta 2+ ornek var → `stratify=y`
- `train_test_split(test_size=0.2, random_state=42)`
- Cikti: `X_train` (120), `X_test` (30), `y_train`, `y_test`

> **Kritik:** Split burada, preprocessing'den **once** yapiliyor.
> Sizinti onleminin birinci katmani bu siralamadir.

### Adim 3 — Iterasyon dongusu baslar

`orchestrator.py:67-120`

```python
for i in range(1, max_iterasyon + 1):        # 1, 2, 3
    state.iterasyon = i
    state.strateji = STRATEJILER[min(i-1, 2)]  # "varsayilan" / "pca_ters" / "imputation_degis"
```

### Adim 4 — DENEME grubu (her iterasyonda tekrar)

`orchestrator.py:73-76`

**4a. `PlannerAgent.run()`** (`planner.py:200-231`)

1. Once **kural-tabanli** `plan(state)` calisir (`planner.py:13-91`):
   - Her kolon icin drop kurallari (`:25-47`): hedef mi? null > %50 mi?
     text/datetime mi? kategorik ve essiz > 50 mi?
   - iris'te hicbiri tetiklenmez → 4 kolon `numeric_cols`'a gider
   - PCA karari (`:49`): `len(numeric_cols) > 10` → 4 > 10 **yanlis**,
     PCA kapali
   - Strateji katmani (`:57-77`): `strateji == "varsayilan"` oldugu icin
     hicbir sey degismez
2. `_gecmis_ozeti(p)` (`:139-177`) → `memory/store.benzer_runlar()`
   cagrilir, benzer gecmis run'lar bulunur
3. `llm.is_available()` (`llm.py:19-27`) — `ANTHROPIC_API_KEY` yoksa
   `False` → LLM cagrilmaz, gecmis bilgisi nota yazilir (`:219-221`)
4. `notes`'a `"kural-tabanli plan kullanildi"` eklenir (`:228`)
- Cikti: `state.plan` doldu (`PreprocessingPlan`)

**4b. `PreprocessorAgent` → `preprocessor.preprocess()`**
(`preprocessor.py:17-53`)

- Sayisal dal: `SimpleImputer(median)` → `StandardScaler`
- Kategorik dal: `SimpleImputer(most_frequent)` → `OneHotEncoder`
- `ColumnTransformer(remainder="drop")` ile birlestirilir
- `use_pca` kapali → PCA adimi eklenmez

```python
state.X_train_t = preprocessor.fit_transform(state.X_train)  # fit SADECE train
state.X_test_t  = preprocessor.transform(state.X_test)       # test'e sadece transform
```

> **Kritik:** Sizinti onleminin ikinci katmani bu iki satirdir
> (`preprocessor.py:50-51`).

- Cikti: `X_train_t` (120,4), `X_test_t` (30,4), `state.preprocessor`

**4c. `ModelerAgent.run()`** (`modeler.py:149-172`)

1. `_candidate_models("classification", 150)` (`modeler.py:23-41`) →
   `{Baseline, LogisticRegression, RandomForest, GradientBoosting}`
2. `llm.is_available()` → `False` → `"kural-tabanli model havuzu
   kullanildi"` basilir
3. `train(state)` (`modeler.py:44-91`):
   - `scoring = "f1_weighted"` (siniflandirma, `:55`)
   - Fold sayisi (`:58-60`): `max(2, min(5, 120//2)) = 5`, sonra sinif
     sinirlamasi → 5
   - Her model `cross_val_score` ile denenir, `try/except` icinde
   - En yuksek `cv_mean` kazanir → `LogisticRegression` (cv=0.958)
   - Kazanan tum train setinde `fit` edilir
- Cikti: `state.model`, `state.best_name`, `state.candidates`

**4d. `EvaluatorAgent` → `evaluator.evaluate()`** (`evaluator.py:58-119`)

- `y_pred = model.predict(X_test_t)`
- Siniflandirma → `accuracy`, `f1_weighted` (`:78-81`)
- `_gini_coefficient` (`:17-33`) cagrilir; iris **3 sinifli** oldugu
  icin `None` doner, `gini` metrigi eklenmez
- `permutation_importance` (`:90-92`) — test setinde, `n_repeats=5`
- `_gini_importance` (`:36-55`) — `LogisticRegression`'da
  `feature_importances_` yok → bos dict
- Cikti: `state.result` doldu (`RunResult`)

### Adim 5 — Iterasyon degerlendirmesi

`orchestrator.py:86-120`

- Sonuc `state.gecmis_denemeler`'e kaydedilir (`:91-101`)
- En iyiyse anlik goruntusu saklanir (`:106-108`, `_goruntu_al`)
- `_yeterli_mi("f1_weighted", 0.9333, "classification")` (`:35-40`)
  → esik 0.80, `0.9333 >= 0.80` → **`True`**
- `break` → dongu 1. iterasyonda durur

### Adim 6 — En iyi sonuc geri yuklenir

`orchestrator.py:122-128`

```python
_goruntu_yukle(state, en_iyi)   # son degil, EN IYI iterasyon
```

### Adim 7 — Kayit

`orchestrator.py:130-132` → `RecorderAgent` → `recorder.log()`
(`recorder.py:6-13`) → `memory/logger.save_run()` (`logger.py:9-32`)

- `runs/20260904_HHMMSS/run.json` (makine okunur, 9 anahtar)
- `runs/20260904_HHMMSS/run.txt` (insan okunur ozet)

### Adim 8 — Ekrana raporlama

`orchestrator.py:134-236` — profil, istatistik tablolari, korelasyonlar,
plan, preprocess boyutlari, iterasyon tablosu, model karsilastirma,
test sonucu, feature importance, Gini importance.

### Akis Ozeti

```
CLI → run() → HAZIRLIK[loader→profiler→splitter]
                    ↓
              ┌─ DENEME[planner→preprocessor→modeler→evaluator]
              │        ↓
              │   esik gecildi mi?
              │    hayir → strateji degistir ─┐
              └──────────────────────────────┘
                     evet ↓
              en iyi iterasyonu geri yukle
                     ↓
              recorder → runs/<ts>/
                     ↓
              ekrana rapor
```

---

## Bolum 3: Cekirdek Dosyalar

### `automl/schemas.py` (105 satir)

**Ozet:** Sistemde dolasan tum verinin seklini tanimlar; butun
dosyalarin konustugu ortak dil.

| Tanim | Satir | Ne ise yarar |
|---|---|---|
| `InferredType` | 9 | `Literal["numeric","categorical","text","datetime"]` — tip denetleyici uyusmazligi vermesin diye tek yerde |
| `TaskType` | 10 | `Literal["classification","regression","clustering"]` |
| `_Strict` | 13-15 | `extra="forbid"` tabani — yazim hatasi olan alan sessizce yutulmaz |
| `ColumnProfile` | 17-35 | Tek kolonun profili: ad, dtype, tespit edilen tip, essiz sayi, null orani + istatistikler |
| `DataProfile` | 37-45 | Veri setinin tamami: satir/kolon sayisi, hedef, task tipi, kolonlar, sinif dengesi, korelasyonlar |
| `PreprocessingPlan` | 48-58 | Plan: hangi kolon nereye, imputation, scaling, encoding, PCA, notlar |
| `ModelScore` | 60-63 | Bir modelin CV skoru (ad, ortalama, std) |
| `ModelSecimi` | 65-68 | LLM'in onerdigi model listesi (sadece LLM cevabini valide etmek icin) |
| `RunResult` | 70-78 | Sonuc: task, adaylar, kazanan model, metrikler, iki importance |
| `RunState` | 80-103 | **Tum ajanlar arasinda dolasan canta** — 21 alan |

**Bagimlilik:** Hicbir proje dosyasini import etmez (yaprak dugum).
**Buna bagimli:** Neredeyse her sey — 8 ajan, orchestrator, testler.

**Dikkat cekici:**
- `_Strict` tabani `ColumnProfile` ve `DataProfile`'da kullaniliyor ama
  `PreprocessingPlan` ve `ModelSecimi`'de **kullanilmiyor** — bu
  bilincli: LLM'den gelen fazladan alanlar hata vermesin, sessizce
  yok sayilsin diye.
- `RunState` bir `@dataclass`, digerleri pydantic `BaseModel`. Mutable
  varsayilanlar `field(default_factory=list)` ile veriliyor
  (dataclass'ta `= []` yazmak hata verir).

**Tutarsizlik (zararsiz):**
- Satir 7-8'deki yorum `steps.py` dosyasindan bahsediyor ama **boyle
  bir dosya yok** — eski bir isimden kalma.
- Satir 72: `RunResult` bir pydantic modeli ama `dataclasses.field()`
  kullaniyor. Test ettim, **calisiyor** (pydantic bunu tanıyor ve `[]`
  uretiyor, ornekler arasinda paylasilmiyor) ama stil tutarsizligi —
  digerleri gibi `= []` yeterdi.

---

### `automl/orchestrator.py` (253 satir)

**Ozet:** Ajanlari sirayla calistirir, self-improvement dongusunu
yonetir ve sonucu ekrana raporlar. Programin giris noktasi.

| Tanim | Satir | Ne ise yarar |
|---|---|---|
| `HAZIRLIK` | 17 | `[loader, profiler, splitter]` — **bir kez** calisir |
| `DENEME` | 19-20 | `[planner, preprocessor, modeler, evaluator]` — **her iterasyonda** |
| `RECORDER` | 21 | Dongu bitince bir kez |
| `ESIKLER` | 24 | `{"classification": 0.80, "regression": 0.50}` |
| `STRATEJILER` | 27 | `["varsayilan", "pca_ters", "imputation_degis"]` |
| `_GORUNTU_ALANLARI` | 30-32 | En iyi iterasyonu geri yuklemek icin saklanan 10 alan |
| `_yeterli_mi()` | 35-40 | Skor esigi gecti mi? Girdi: metrik adi, deger, task. Cikti: `bool` |
| `_goruntu_al()` | 43-45 | Girdi: `state`. Cikti: 10 alanlik `dict` (anlik goruntu) |
| `_goruntu_yukle()` | 48-51 | Anlik goruntuyu `state`'e geri yazar |
| `run()` | 54-236 | Ana fonksiyon. Girdi: veri yolu, hedef, max iterasyon. Cikti: `RunState` |

**Bagimlilik:** 8 ajan + `schemas`.
**Buna bagimli:** `report.py`, `tests/test_pipeline.py`,
`tests/test_no_leakage.py` (sadece `HAZIRLIK`/`DENEME` siralamasi icin).

**Dikkat cekici — HAZIRLIK/DENEME ayrimi (`:15-20`):**
Sadece performans icin degil. Split her iterasyonda tekrarlansaydi
**her denemede farkli bir test seti** olusur ve iterasyonlarin skorlari
karsilastirilamaz hale gelirdi. Bu ayrim dogruluk gerekcelidir.

**Dikkat cekici — sonsuz dongu korumasi (uc katman):**
1. `for i in range(1, max_iterasyon + 1)` — sabit ust sinir (`:67`)
2. Esik gecilince `break` (`:113`)
3. Bir iterasyon patlarsa `try/except` yakalar, hatayi kaydeder,
   **sonraki iterasyona devam eder** (`:77-84`). Hepsi patlarsa acik
   `RuntimeError` (`:128`).

**Dikkat cekici — beraberlik kurali (`:106`):**
Karsilastirma kesin buyuktur (`>`) ile — esitlikte **ilk** iterasyon
korunur, yani daha basit plan tercih edilir.

**Kucuk sorunlar:**
- `_yeterli_mi()` `metric_name` parametresini aliyor ama **kullanmiyor**
  (`:44`).
- `_f()` yardimci fonksiyonu `for` dongusunun **icinde** tanimli
  (`:155-156`) — her kolonda yeniden olusuyor. Isleyisi etkilemiyor
  ama dongunun disina alinmali.
- `run()` fonksiyonu ~180 satir; dongu ve raporlama ayrilabilirdi.

---

### `automl/llm.py` (72 satir)

**Ozet:** LLM cagrisi icin tek arayuz; **hicbir kosulda istisna
firlatmaz**, ya gecerli nesne ya `None` dondurur.

| Tanim | Satir | Ne ise yarar |
|---|---|---|
| `MODEL` | 13 | `"claude-sonnet-4-5"` |
| `MAX_TOKENS` | 14 | `8192` |
| `T` | 16 | `TypeVar` — `ask_json`'un donus tipini cagirana gore belirler |
| `is_available()` | 19-27 | Anahtar **ve** paket var mi? Cikti: `bool` |
| `_temizle()` | 30-39 | ` ```json ... ``` ` sarmalayicisini soyar. Girdi/Cikti: `str` |
| `ask_json()` | 42-72 | LLM'e sor → JSON parse → sema dogrula. Girdi: system prompt, user prompt, pydantic sinifi. Cikti: nesne veya `None` |

**Bagimlilik:** `pydantic` + (fonksiyon icinde) `anthropic`. **Hicbir
proje dosyasini import etmez.**
**Buna bagimli:** `agents/planner.py`, `agents/modeler.py`.

**Dikkat cekici — `anthropic` importu fonksiyon icinde (`:24`, `:53`):**
Modul seviyesinde `import anthropic` yazilsaydi paket kurulu olmayan
bir ortamda **tum sistem cokerdi**. Fonksiyon icine alinca opsiyonel
bagimlilik gercekten opsiyonel oluyor. (Ayrica `try/except ImportError`
ile modul seviyesinde denendi ama pyright "possibly unbound" uyarisi
verdigi icin bu yol secildi.)

**Dikkat cekici — tek `except Exception` (`:70-72`):**
Ag hatasi, 401, bozuk JSON, sema hatasi — hepsi ayni yere duser ve
`None` doner. Boylece **cagiran taraf tek bir sey kontrol eder:**
`None` mi degil mi.

---

## Bolum 4: Ajanlar

Hepsi `Agent` arayuzunu uygular: `RunState` al → `RunState` dondur.
Hicbiri digerini import etmez; sadece `state` uzerinden konusurlar.

### `agents/base.py` (16 satir)

**Ozet:** Tum ajanlarin ortak arayuzu.

- `Agent` (ABC): `name` sinif degiskeni, soyut `run(state)` metodu,
  `__call__(state)` → `run(state)` (satir 15-16).
- `__call__` sayesinde orchestrator `agent(state)` yazabiliyor.

**Bagimlilik:** `schemas`. **Buna bagimli:** 8 ajanin hepsi.

---

### `agents/loader.py` (18 satir) — 1. adim

**Ozet:** CSV'yi diskten okur. Sistemin en kucuk dosyasi.

- `load(state)` (`:7-9`): `state.df = pd.read_csv(state.data_path)`
- `LoaderAgent` (`:12-18`): `name = "loader"`

**Girdi:** `state.data_path` · **Cikti:** `state.df`

**Dikkat cekici (olumsuz):** `parse_dates` **kullanilmiyor**. Bu
yuzden CSV'deki tarih kolonlari string olarak okunuyor ve `profiler`
tarafindan `datetime` degil `text` olarak isaretleniyor. Sonuc ayni
(atiliyorlar) ama mekanizma beklenenden farkli.

---

### `agents/profiler.py` (167 satir) — 2. adim

**Ozet:** Veriyi tanir — her kolonun tipini, hedefi, gorev tipini,
istatistikleri ve korelasyonlari cikarir. **Domain-bagimsizligin
kalbi.**

| Fonksiyon | Satir | Girdi → Cikti |
|---|---|---|
| `_detect_type()` | 14-42 | `(Series, n_rows)` → `"numeric"/"categorical"/"text"/"datetime"` |
| `_numeric_stats()` | 44-59 | `Series` → `dict` (ort/std/min/q25/medyan/q75/max/carpiklik) |
| `_categorical_stats()` | 62-72 | `Series` → `dict` (en sik deger, orani) |
| `_high_correlations()` | 75-91 | `(df, kolonlar, esik=0.85)` → `[(a, b, r), ...]` ilk 15 |
| `profile()` | 93-158 | `RunState` → `RunState` (`state.profile` dolar) |
| `ProfilerAgent` | 161-167 | `name = "profiler"` |

**Girdi:** `state.df`, `state.target` · **Cikti:** `state.profile`

**Dikkat cekici — oransal tip esigi (`:38-41`):**

```python
esik = max(2, min(20, int(n_rows * 0.05)))
return "categorical" if clean.nunique() <= esik else "numeric"
```

Sabit sayi degil **oran** kullaniliyor cunku bir tamsayi kolonunun
kategori mi olcum mu oldugu veri boyutuna gore degisir. 1000 satirda
8 essiz deger = kategori kodu; 40 satirda 8 essiz deger = sayisal olcum.
- `max(2, ...)`: 20 satirlik veride esik 1 cikardi ve ikili hedef
  yanlislikla sayisal sayilirdi (task tespiti bozulurdu).
- `min(20, ...)`: 100.000 satirda esik 5000 cikardi ve one-hot
  encoding bellegi patlatirdi.

**Dikkat cekici — ondalik testi (`:34-35`):** Bir kolonda tek bir
ondalikli deger varsa kesin sayisaldir. Ucuz ve yanlis pozitif
uretmedigi icin esik kuralindan **once** calisir.

**Dikkat cekici — metin ayrimi (`:21-24`):** `essiz > 50` **VE**
`essiz/satir > 0.5` — iki kosul birlikte. Tek basina birincisi
yetersizdi: 10.000 satirda 60 sehir adi gecerli bir kategoriktir.

**Kucuk sorun:** Satir 2'de `import numpy as np` var ama **dosyada
hic kullanilmiyor** — silinebilir.

---

### `agents/splitter.py` (48 satir) — 3. adim

**Ozet:** Train/test ayrimi. Preprocessing'den **once** calisir.

- `split(state)` (`:7-38`), `SplitterAgent` (`:41-47`)

**Girdi:** `state.df`, `state.profile`
**Cikti:** `X_train`, `X_test`, `y_train`, `y_test`

**Dikkat cekici — kosullu stratify (`:22-24`):** Siniflandirmada ve
en az kalabalik sinifta 2+ ornek varsa `stratify=y`. Az ornekli sinifin
test setinden tamamen kaybolmasini onler.

**Dikkat cekici — cift katmanli koruma (`:26-34`):** Stratify
`ValueError` verirse (cok kucuk veri) stratify'siz tekrar denenir.
`sample.csv` (6 satir) bu sayede cokmuyor.

**Dikkat cekici — hedefsiz durum (`:14-17`):** `target is None` ise
tum veri `X_train`, `X_test` bos dondurulur (clustering yolu).

---

### `agents/planner.py` (231 satir) — 4. adim

**Ozet:** Profile bakarak preprocessing planini uretir. Kural-tabanli
cekirdek + strateji katmani + LLM katmani + gecmis run baglami.

| Fonksiyon | Satir | Girdi → Cikti |
|---|---|---|
| `SYSTEM_PROMPT` | 7-10 | LLM'e verilen rol tanimi |
| `plan()` | 13-91 | `RunState` → `RunState` (kural-tabanli, **LLM'siz**) |
| `_profil_ozeti()` | 94-136 | `(DataProfile, plan)` → LLM prompt metni |
| `_gecmis_ozeti()` | 139-177 | `DataProfile` → `(prompt parcasi, not satirlari)` |
| `_kolonlar_gecerli()` | 180-192 | `(oneri, profile)` → `bool` (uydurma kontrolu) |
| `PlannerAgent.run()` | 200-231 | 4 adimli akis: kural → gecmis → LLM → secim |

**Girdi:** `state.profile`, `state.strateji` · **Cikti:** `state.plan`

**Dikkat cekici — planner hicbir kolon adi bilmez:** Dosyada tek bir
somut kolon adi gecmez. Kararlar sadece profildeki **ozelliklere**
dayanir.

**Drop kurallari (`:25-47`), sirayla:**

| Kural | Satir | Gerekce |
|---|---|---|
| Hedefin kendisi atlanir | 26-27 | Ozellik olamaz, sizinti olur |
| `null_ratio > 0.5` | 29-32 | Yarisindan fazlasi bos; imputation gercek bilgi degil uydurma uretir |
| `text` / `datetime` | 34-37 | NLP ve tarih ozellik cikarimi yok |
| `categorical` ve `n_unique > 50` | 39-42 | One-hot patlamasi + asiri ogrenme riski |

**Dikkat cekici — PCA karari (`:49`):** `len(numeric_cols) > 10 AND
n_rows > 50`. Iki kosul birlikte: az kolonda PCA'nin kazanci yok;
az satirda kovaryans tahmini gurultulu olur.

**Dikkat cekici — hedef sizmasi korumasi (`:182`):**

```python
gecerli = {c.name for c in p.columns if c.name != p.target}
```

Gecerli kume **hedef haric**. LLM hedefi ozellik listesine koyarsa
plan reddedilir — hem sizinti hem cokme onlenir.

**Dikkat cekici — LLM yoksa gecmis sadece nota yazilir (`:219-221`):**
Yani "gecmisten ogrenme" karari **ancak LLM acikken** etkiler; kapaliyken
bilgi sadece gorunurdur.

---

### `agents/preprocessor.py` (62 satir) — 5. adim

**Ozet:** Plani sklearn `Pipeline`'a cevirir ve uygular. Veri sizintisi
onleminin ikinci katmani burada.

- `_make_onehot()` (`:12-14`): `OneHotEncoder(handle_unknown="ignore",
  sparse_output=False)`
- `preprocess(state)` (`:17-53`), `PreprocessorAgent` (`:56-62`)

**Girdi:** `state.plan`, `X_train`, `X_test`
**Cikti:** `X_train_t`, `X_test_t`, `state.preprocessor`

**Dikkat cekici — fit/transform ayrimi (`:49-51`):**

```python
# KRITIK: fit sadece train'de, test'e sadece transform
state.X_train_t = preprocessor.fit_transform(state.X_train)
state.X_test_t  = preprocessor.transform(state.X_test)
```

Tum sizinti garantisi bu iki satira dayanir. `tests/test_no_leakage.py`
bunu dogrudan test eder.

**Dikkat cekici — `handle_unknown="ignore"` (`:14`):** Test setinde
egitimde gorulmemis bir kategori cikarsa sistem cokmez, sifir vektorune
donusur.

**Dikkat cekici — `remainder="drop"` (`:39`):** Plana girmeyen kolonlar
otomatik dusurulur; `drop_cols` listesini ayrica uygulamaya gerek kalmaz.

---

### `agents/modeler.py` (163 satir) — 6. adim

**Ozet:** Aday modelleri CV ile karsilastirir, kazanani egitir.

| Fonksiyon | Satir | Girdi → Cikti |
|---|---|---|
| `SYSTEM_PROMPT` | 17-20 | LLM rol tanimi |
| `_candidate_models()` | 23-41 | `(task_type, n_rows)` → `{ad: model}` sozlugu |
| `train()` | 44-91 | `(RunState, izinli_modeller=None)` → `RunState` |
| `_model_ozeti()` | 94-129 | `(RunState, havuz)` → LLM prompt metni |
| `_secim_gecerli()` | 132-141 | `(ModelSecimi, havuz)` → `bool` |
| `ModelerAgent.run()` | 149-172 | Havuzu kur → LLM'e sor → `train()` |

**Girdi:** `X_train_t`, `y_train`, `profile.task_type`
**Cikti:** `state.model`, `state.best_name`, `state.candidates`

**Aday havuz (`:23-41`):** Task basina 4 model — bir baseline, bir
dogrusal, bir bagging, bir boosting. Cesitlilik bilincli: farkli veri
yapilarini kapsar.

**Dikkat cekici — baseline neden var:** Baseline olmadan bir skor
anlamsizdir. %95 accuracy, veri %95 tek siniftansa hicbir sey
ogrenmemis demektir. `cancer`'da baseline 0.482 → kazanan 0.980;
aradaki fark gercekten ogrenilen bilgidir.

**Dikkat cekici — otomatik fold sayisi (`:58-60`):**

```python
n_splits = max(2, min(5, len(y) // 2))
if p.task_type == "classification":
    n_splits = max(2, min(n_splits, int(y.value_counts().min())))
```

Iki asamali: genel sinir + siniflandirmada "en az kalabalik sinifin
ornek sayisini gecemez". Sabit `cv=5` olsaydi 6 satirlik `sample.csv`
cokerdi.

**Dikkat cekici — model basina `try/except` (`:63-77`):** Bir model
patlarsa atlanir, digerleri devam eder. Hicbiri calismazsa acik hata.

**Kucuk sorun:** `_candidate_models(task_type, n_rows)` — `n_rows`
parametresi **hic kullanilmiyor** (`:23`). Muhtemelen veri boyutuna
gore havuz degistirmek planlanmis ama yapilmamis.

---

### `agents/evaluator.py` (~120 satir) — 7. adim

**Ozet:** Test setinde degerlendirir; metrikler, Gini katsayisi ve iki
tur feature importance uretir.

| Fonksiyon | Satir | Girdi → Cikti |
|---|---|---|
| `_gini_coefficient()` | 17-33 | `(model, X_te, y_te)` → `float` veya `None` |
| `_gini_importance()` | 36-55 | `(model, feat_names)` → `dict` (bos olabilir) |
| `evaluate()` | 58-119 | `RunState` → `RunState` (`state.result` dolar) |
| `EvaluatorAgent` | — | `name = "evaluator"` |

**Girdi:** `state.model`, `X_test_t`, `y_test` · **Cikti:** `state.result`

**Metrikler:** Regresyon → `r2`, `rmse`, `mae` (ana: `r2`).
Siniflandirma → `accuracy`, `f1_weighted` (ana: `f1_weighted`),
+ ikili ise `gini`.

**Dikkat cekici — Gini katsayisi (`:17-33`):** `2 * AUC - 1`. Uc
kosullu koruma: sinif sayisi 2 degilse `None`, `predict_proba` yoksa
`None`, hata olursa `None`. iris 3 sinifli oldugu icin `gini` cikmaz;
cancer 2 sinifli oldugu icin cikar (0.9907).

**Dikkat cekici — Gini importance (`:36-55`):** `feature_importances_`
yoksa bos dict; deger sayisi ozellik sayisiyla uyusmuyorsa bos dict.
Yani **sadece agac modellerinde** dolar.

**Dikkat cekici — neden ana metrik permutation:** `permutation_importance`
model-agnostik (her modelde calisir) ve **test setinde** olculur.
Gini importance sadece agaclarda var ve egitim verisine bakar.
REPORT.md'de 7 veri setinin 4'unde dogrusal model kazaniyor — ana
metrik Gini olsaydi bu vakalarda tablo bos kalirdi.

---

### `agents/recorder.py` (22 satir) — 8. adim

**Ozet:** Run'i diske kaydeder.

- `log(state)` (`:6-13`): `save_run(state, sure)` cagirir, klasoru basar
- `RecorderAgent` (`:16-22`)

**Girdi:** Tum `state` · **Cikti:** `runs/<timestamp>/` klasoru

**Dikkat cekici — fonksiyon ici import (`:8`):**
`from automl.memory.logger import save_run` fonksiyonun **icinde**.
Modul seviyesinde olsaydi `agents` → `memory` bagimliligi her import'ta
yuklenirdi; bu sekilde sadece gercekten kaydederken yukleniyor.

**Dikkat cekici — `getattr` ile guvenli erisim (`:10`):**
`getattr(state, "sure_sn", 0.0)` — alan yoksa cokmek yerine 0.0.

---

### `agents/__init__.py` (22 satir)

**Ozet:** Tum ajanlari ve fonksiyonlari toplu olarak dışa acar
(`__all__` ile 17 isim).

**KULLANILMIYOR.** Projede hicbir dosya `from automl.agents import ...`
yazmiyor; orchestrator ve testler dogrudan alt modulleri import
ediyor (`from automl.agents.loader import LoaderAgent`). Zararsiz ama
her `automl.agents.*` import'unda 8 ajanin tamamini (ve dolayisiyla
sklearn'un buyuk kismini) yukluyor.

---

## Bolum 5: Hafiza Katmani

### `automl/memory/logger.py` (78 satir)

**Ozet:** Bir run'i `runs/<timestamp>/` altina JSON + metin olarak yazar.

| Fonksiyon | Satir | Girdi → Cikti |
|---|---|---|
| `save_run()` | 9-32 | `(state, sure_sn)` → yazilan klasorun `Path`'i |
| `_ozet()` | 35-78 | `dict` → insan okunur metin |

**`run.json` icerigi (9 anahtar, `:15-26`):** `timestamp`, `data_path`,
`sure_sn`, `profile`, `plan`, `result`, `iterasyonlar`,
`en_iyi_iterasyon`, `en_iyi_strateji`.

**Bagimlilik:** Hicbir proje dosyasini import etmez (sadece `json`,
`datetime`, `pathlib`). **Buna bagimli:** `agents/recorder.py`.

**Dikkat cekici — `getattr` ile geriye donuk uyum (`:23-25`):**
`getattr(state, "gecmis_denemeler", [])` — eski `RunState`'lerde bu
alanlar yoktu, `getattr` sayesinde eski kayitlar bozulmuyor.

**Dikkat cekici — iki format:** `run.json` makine icin (store.py okur),
`run.txt` insan icin. Ayni bilgi, iki hedef kitle.

---

### `automl/memory/store.py` (57 satir)

**Ozet:** Gecmis run'lari okur ve mevcut profile benzeyenleri bulur.

| Fonksiyon | Satir | Girdi → Cikti |
|---|---|---|
| `tum_runlar()` | 8-19 | — → `list[dict]`, yeniden eskiye sirali |
| `_benzerlik()` | 22-42 | `(profil1, profil2)` → `float` 0.0-1.0 |
| `benzer_runlar()` | 45-57 | `(profile, esik=0.6, limit=5)` → benzer run listesi |

**Benzerlik skoru — uc sinyal (`:22-42`):**

| Sinyal | Puan | Mantik |
|---|---|---|
| Ayni task tipi | 0.4 | Farkliysa **0.0 doner** (kapi kosulu) |
| Satir sayisi 2 kat icinde | +0.3 | 150 satirdan ogrenilen 100.000'e uymaz |
| Sayisal kolon orani < 0.25 fark | +0.3 | Farkli yapi, farkli preprocessing ister |

Esik 0.6 demek: ayni task (0.4) tek basina **yetmez**, en az bir
yapisal sinyalin de tutmasi gerekir.

**Bagimlilik:** Hicbir proje dosyasini import etmez.
**Buna bagimli:** `agents/planner.py`, `show_runs.py`.

**Dikkat cekici — bozuk dosya toleransi (`:15-18`):** Her `run.json`
ayri `try/except` icinde okunur; bozuk bir dosya tum gecmisi
kullanilamaz hale getirmez, sadece atlanir.

---

## Bolum 6: Testler

30 test, `python3 -m pytest tests/ -v` ile ~1.8 saniyede gecer.

### `tests/conftest.py` (11 satir)

**Ozet:** Tum testlerde otomatik calisan tek fixture.

- `llm_kapali` (autouse): `ANTHROPIC_API_KEY`'i siler.

**Neden:** Testler LLM'e cikmasin — deterministik, ucretsiz ve
internetsiz calissinlar. Kural-tabanli yol her zaman calistigi icin
testler onu dogrular.

### `tests/test_type_detection.py` (70 satir, 9 test)

**Neyi koruyor:** Tip tespitinin her dali — domain-bagimsizligin temeli.

Ondalikli→numeric, az kardinaliteli int→categorical, cok kardinaliteli
int→numeric, string→categorical, her satiri farkli string→text,
bool→categorical, gercek datetime→datetime, **esigin veri boyutuna
gore degismesi**, tamamen bos kolon cokmuyor.

**En degerlisi** `test_esik_veri_boyutuna_gore_degisir` (`:52-66`):
**ayni** Series, `n_rows=1000` iken categorical, `n_rows=40` iken
numeric. Sabit esige donulurse bu test kirmizi olur.

### `tests/test_task_detection.py` (69 satir, 5 test)

**Neyi koruyor:** Hedef kolon ve gorev tipi tespiti.

Sayisal hedef→regression, kategorik hedef→classification, hedef
verilmezse son kolon, olmayan hedef→`ValueError` (mesajda hem eksik
kolon adi hem mevcut kolonlar), profile olmadan plan→`RuntimeError`.

### `tests/test_no_leakage.py` (111 satir, 5 test) — EN KRITIK

**Neyi koruyor:** Veri sizintisi. Sizinti hicbir hata mesaji vermez,
sadece skoru sisirir — bu yuzden otomatik test sart.

1. `test_scaler_test_setinden_etkilenmiyor` (`:22-45`): Test setine
   egitim verisinin ~20.000 kati uc degerler konur; scaler'in ogrendigi
   ortalamanin **49.5'te kaldigi** dogrulanir.
2. `test_test_setine_sadece_transform_uygulaniyor` (`:48-60`):
   Donusturulmus train ortalamasi ~0, test ortalamasi > 5.
3. `test_bos_deger_imputation_da_traindan_ogreniliyor` (`:63-75`):
   Sizinti scaler'da degil **imputer'da** da olabilir.
4. `test_split_preprocessten_once_calisiyor` (`:78-87`): Yapisal test —
   `HAZIRLIK + DENEME` icinde `splitter` indeksi `preprocessor`'dan
   kucuk olmali.
5. `test_uctan_uca_scaler_sadece_traini_gormus` (`:90-111`): Gercek
   veride scaler ortalamasi `X_train`'e esit **ve** tum veriye esit
   **degil**. Ikinci assert kritik: esit olsaydi test hicbir sey
   kanitlamazdi.

### `tests/test_pipeline.py` (90 satir, 11 test)

**Neyi koruyor:** `messy.csv` uzerinde uctan uca dayaniklilik ve drop
kararlari.

`messy_state` fixture (`:14-22`) module-scope — boru hatti bir kez
calisir, 11 test ayni sonuca bakar.

Cokmuyor, task classification, `bos_kolon`/`urun_kodu`/`aciklama`/
`kayit_tarihi` atildi, kullanilabilir kolonlar korundu, %15 eksikli
kolon atilmadi, sonuc uretildi, iterasyon kaydi tutuldu,
**donusturulmus matriste NaN kalmadi**.

> **Test kapsami bosluklari** icin Bolum 9 sonuna bakin.

---

## Bolum 7: Yardimci Scriptler

### `make_data.py` (89 satir)

**Ozet:** 5 test veri setini uretir (internet gerektirmez).

| Fonksiyon | Satir | Ne yapar |
|---|---|---|
| `kaydet()` | 10-16 | sklearn loader'indan CSV uretir |
| `kaydet_nonlinear()` | 24-42 | `make_classification` ile dogrusal olmayan veri |
| `kaydet_messy()` | 44-88 | Eksik deger + karisik tip iceren zorlu veri |

Uretilenler: `iris.csv` (150x5), `cancer.csv` (569x31),
`diabetes.csv` (442x11), `nonlinear.csv` (800x13), `messy.csv` (400x9).

**Dikkat cekici — `nonlinear.csv` (`:26-34`):** `n_clusters_per_class=3`
ve `class_sep=0.7` ile bilincli olarak **dogrusal olmayan** uretildi —
agac modellerinin kazandigini gostermek icin. Calisti: RandomForest
cv=0.753, LogisticRegression cv=0.658.

**Dikkat cekici — `messy.csv` (`:44-88`):** Planner'in dort drop
kuralinin **her birini** tetikleyecek sekilde tasarlandi: `bos_kolon`
(%100 null), `urun_kodu` (60 essiz), `aciklama` (serbest metin),
`kayit_tarihi` (tarih). Ayrica `olcum_a` %15, `seviye` %8 eksikli —
bunlar **atilmamali**, imputation ile kullanilmali.

**Bagimlilik:** Proje dosyasi import etmez (sadece pandas, numpy,
sklearn). Bagimsiz calisir.

---

### `report.py` (103 satir)

**Ozet:** `data/*.csv` altindaki tum veri setlerini sirayla boru
hattindan gecirip `REPORT.md` uretir.

| Fonksiyon | Satir | Girdi → Cikti |
|---|---|---|
| `_satir()` | 20-51 | `Path` → rapor satiri `dict` (veya `{"hata": ...}`) |
| `main()` | 54-100 | — → `REPORT.md` yazar |

**Dikkat cekici — cikti bastirma (`:26-27`):**
`contextlib.redirect_stdout(buf)` ile boru hattinin kendi ciktisi
yutulur; sadece ozet satiri basilir.

**Dikkat cekici — hata toleransi (`:28-33`):** Bir veri seti patlarsa
tablo satirinda `HATA:` yazar ve **digerleri devam eder**.

**Bagimlilik:** `automl.orchestrator`.

---

### `show_runs.py` (16 satir)

**Ozet:** Gecmis run'lari tek satirlik ozetlerle listeler.

Modul seviyesinde calisan bir script (fonksiyon yok,
`if __name__ == "__main__"` **yok** — import edilirse hemen calisir).
`tum_runlar()` cagirir, `profile` veya `result` eksik olanlari atlar.

**Bagimlilik:** `automl.memory.store`.

---

## Bolum 8: Dokumanlar ve Yapilandirma

| Dosya | Satir | Kime hitap ediyor | Icerik |
|---|---|---|---|
| `README.md` | 204 | **Kullanici / gelistirici** | Ne yapar, kurulum, kullanim komutlari, mimari diyagram, domain-bagimsizlik, sizinti onlemi, LLM katmani, self-improvement, bilinen sinirlar |
| `SUNUM.md` | 1665 | **Hoca / juri** | 17 bolum: iddia, mimari gerekceleri, her bilesen, domain-bagimsizlik detayi, istatistik, sizinti, model secimi, PCA+Gini, LLM, self-improvement, hafiza, testler, sonuclar, odev kavramlari tablosu, sinirlar, demo akisi, 18 soru-cevap |
| `REPORT.md` | 15 | **Kanit belgesi** | `report.py` tarafindan **uretilir**, elle duzenlenmez. 7 veri setinin karsilastirmali tablosu |
| `DOSYA-REHBERI.md` | — | **Gelistirici (sen)** | Bu belge — dosya dosya hizli referans |
| `requirements.txt` | 12 | **Kurulum** | pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, pydantic 2.13.4, anthropic 1.3.0, pytest 9.1.1, pyright 1.1.411 |
| `.gitignore` | 8 | **Git** | `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `runs/`, `data/*.csv`, `.DS_Store`, `.vscode/` |

**Belgeler arasi is bolumu:** `README.md` "nasil kullanirim",
`SUNUM.md` "neden boyle yaptim", `REPORT.md` "iste kanit",
`DOSYA-REHBERI.md` "hangi dosyada ne var".

**`.gitignore` notu:** `runs/` ve `data/*.csv` git disinda. Yani depoyu
klonlayan biri once `python3 make_data.py` calistirmali. Su an git'te
**30 dosya** takip ediliyor.

### `data/` klasoru

7 CSV, `.gitignore`'da:

| Dosya | Boyut | Alan | Hedef |
|---|---|---|---|
| `iris.csv` | 150x5 | Botanik | `tur` (3 sinif) |
| `cancer.csv` | 569x31 | Tibbi teshis | `teshis` (2 sinif) |
| `diabetes.csv` | 442x11 | Tibbi ilerleme | `ilerleme` (surekli) |
| `nonlinear.csv` | 800x13 | Sentetik | `hedef` (2 sinif) |
| `messy.csv` | 400x9 | Sentetik (kirli) | `hedef` (2 sinif) |
| `sample.csv` | 6x4 | Perakende | `satin_aldi` (2 sinif) |
| `sample_reg.csv` | 8x4 | Emlak | `fiyat` (surekli) |

`sample.csv` ve `sample_reg.csv` `make_data.py` tarafindan
**uretilmiyor** — elle olusturulmus, Turkce kolon adli
(`yas/maas/sehir`, `metrekare/oda_sayisi/ilce`) kucuk veri setleri.
Silmeyin: kucuk veri dayanikliligini ve farkli domain'i kanitliyorlar.

### `runs/` klasoru

Su an **50 klasor**. Her biri `<YYYYMMDD_HHMMSS>/` formatinda ve icinde
`run.json` + `run.txt`. `.gitignore`'da. Her calistirma bir klasor
ekler — `report.py` bir seferde 7 tane ekler. Periyodik temizlik
gerekebilir (`benzer_runlar` her planner cagrisinda hepsini okuyor).

---

## Bolum 9: Bagimlilik Haritasi

### Proje ici import tablosu

| Dosya | Import ettikleri (proje ici) |
|---|---|
| `automl/__init__.py` | — (bos) |
| `automl/schemas.py` | — |
| `automl/llm.py` | — |
| `automl/memory/logger.py` | — |
| `automl/memory/store.py` | — |
| `automl/agents/base.py` | `schemas` |
| `automl/agents/loader.py` | `agents.base`, `schemas` |
| `automl/agents/profiler.py` | `agents.base`, `schemas` |
| `automl/agents/splitter.py` | `agents.base`, `schemas` |
| `automl/agents/preprocessor.py` | `agents.base`, `schemas` |
| `automl/agents/recorder.py` | `agents.base`, `schemas`, `memory.logger` *(fonksiyon ici)* |
| `automl/agents/planner.py` | `llm`, `agents.base`, `memory.store`, `schemas` |
| `automl/agents/modeler.py` | `llm`, `agents.base`, `schemas` |
| `automl/agents/evaluator.py` | `agents.base`, `schemas` |
| `automl/agents/__init__.py` | 9 ajan modulu **(kimse bunu import etmiyor)** |
| `automl/orchestrator.py` | 8 ajan + `schemas` |
| `report.py` | `orchestrator` |
| `show_runs.py` | `memory.store` |
| `make_data.py` | — |
| `tests/test_type_detection.py` | `agents.profiler` |
| `tests/test_task_detection.py` | `agents.profiler`, `schemas`, `agents.planner` *(fonksiyon ici)* |
| `tests/test_no_leakage.py` | `agents.preprocessor`, `schemas`, `orchestrator`/`loader`/`planner`/`profiler`/`splitter` *(fonksiyon ici)* |
| `tests/test_pipeline.py` | `orchestrator` |

### Katman diyagrami

```
  KATMAN 0 (yaprak — hicbir proje dosyasi import etmez)
  ┌──────────┐  ┌────────┐  ┌───────────┐  ┌──────────┐
  │schemas.py│  │ llm.py │  │mem/store  │  │mem/logger│
  └────┬─────┘  └───┬────┘  └─────┬─────┘  └────┬─────┘
       │            │             │             │
  KATMAN 1          │             │             │
  ┌────▼─────┐      │             │             │
  │ base.py  │      │             │             │
  └────┬─────┘      │             │             │
       │            │             │             │
  KATMAN 2 (8 ajan) │             │             │
  ┌────▼─────────────▼─────────────▼─────────────▼────┐
  │ loader  profiler  splitter  planner  preprocessor │
  │ modeler  evaluator  recorder                      │
  └────┬──────────────────────────────────────────────┘
       │
  KATMAN 3
  ┌────▼───────────┐
  │ orchestrator.py│
  └────┬───────────┘
       │
  KATMAN 4 (giris noktalari)
  ┌────▼─────┐  ┌─────────────┐  ┌──────────────┐
  │report.py │  │test_pipeline│  │ CLI (-m ...) │
  └──────────┘  └─────────────┘  └──────────────┘

  Bagimsiz: make_data.py, show_runs.py(→store)
```

### Dongusel bagimlilik kontrolu

**Sonuc: DONGU YOK.** AST tabanli tarama ile dogrulandi. Katmanlar
tek yonlu: yaprak → base → ajanlar → orchestrator → giris noktalari.

Dongu riskini onleyen iki tercih:
- `automl/__init__.py` **bos** — `from automl import llm` yazan planner
  ve modeler bir cevrim tetiklemiyor.
- `recorder.py` `memory.logger`'i **fonksiyon icinde** import ediyor.

### Test kapsami bosluklari

Testlerde **dogrudan** hic gecmeyen moduller:

| Modul | Durum |
|---|---|
| `automl/llm.py` | ❗ **Dogrulama katmaninin otomatik testi yok** — `_temizle`, `is_available`, `ask_json` test edilmiyor |
| `agents/planner.py` `_kolonlar_gecerli` | ❗ Uydurma kolon / hedef sizmasi reddi **test edilmiyor** |
| `agents/modeler.py` `_secim_gecerli` | ❗ Uydurma model reddi **test edilmiyor** |
| `memory/store.py` | Test yok (uctan uca dolayli calisiyor) |
| `memory/logger.py` | Test yok (uctan uca dolayli calisiyor) |
| `agents/recorder.py` | Test yok |
| `agents/evaluator.py` | Dogrudan test yok (uctan uca dolayli) |
| `agents/modeler.py` `train()` | Dogrudan test yok (uctan uca dolayli) |

En onemlisi ilk uc satir: **LLM guvenlik dogrulama katmani** —
sistemin en kritik korumalarindan biri — otomatik test edilmiyor.
Bu fonksiyonlar elle dogrulandi ama `tests/` altinda karsiliklari yok.

---

## Bolum 10: "Sunu Degistirmek Istersem Nereye Bakmaliyim?"

| Ne yapmak istiyorsun | Nereye bak | Not |
|---|---|---|
| **Yeni bir model eklemek** | `agents/modeler.py:23-41` `_candidate_models()` | Sozluge `"Ad": ModelNesnesi()` ekle; `train()` otomatik dener |
| **Tip tespit esigini degistirmek** | `agents/profiler.py:38-41` `_detect_type()` | `max(2, min(20, n_rows*0.05))` — degistirirsen `test_type_detection.py:52-66` kirilabilir |
| **Yeni bir drop kurali eklemek** | `agents/planner.py:25-47` `plan()` | Dongu icine `if` blogu ekle, `drop_cols.append()` + `notes.append()` |
| **PCA kosulunu degistirmek** | `agents/planner.py:49-52` | `len(numeric_cols) > 10 and n_rows > 50` |
| **Yeni bir iterasyon stratejisi** | `orchestrator.py:27` + `agents/planner.py:57-77` | **Iki yer:** listeye ad ekle, planner'da `elif` dali yaz |
| **Iterasyon esiklerini degistirmek** | `orchestrator.py:24` `ESIKLER` | `{"classification": 0.80, "regression": 0.50}` |
| **Log formatini degistirmek** | `memory/logger.py:15-26` (JSON) / `35-78` (metin) | Ikisi ayri; JSON'u degistirirsen `store.py` okumasi etkilenebilir |
| **Benzerlik skorunu degistirmek** | `memory/store.py:22-42` `_benzerlik()` | Uc sinyal ve puanlari; esik `:45`'te (0.6) |
| **LLM modelini degistirmek** | `automl/llm.py:13` `MODEL` | Tek satir |
| **LLM prompt'unu degistirmek** | `agents/planner.py:7-10, 94-136` / `agents/modeler.py:17-20, 85-120` | Her ajanin kendi `SYSTEM_PROMPT` + ozet fonksiyonu var |
| **Yeni metrik eklemek** | `agents/evaluator.py:70-85` `evaluate()` | `metrics` sozlugune ekle; `main_metric` ana metrigi belirler |
| **Imputation/scaling stratejisi** | `agents/preprocessor.py:23-31` | `SimpleImputer`/`StandardScaler` burada kuruluyor |
| **Train/test oranini degistirmek** | `agents/splitter.py:27-29` | `test_size=0.2`, `random_state=42` |
| **CV fold sayisi mantigi** | `agents/modeler.py:58-60` | Kucuk veri korumasi burada — dikkatli degistir |
| **Ekran ciktisi/rapor duzeni** | `orchestrator.py:134-236` | Tum `print`'ler burada toplanmis |
| **Yeni bir ajan eklemek** | `agents/yeni.py` + `orchestrator.py:17-20` | `Agent`'tan turet, `HAZIRLIK` veya `DENEME` listesine ekle |
| **Yeni CLI argumani** | `orchestrator.py:238-244` | `argparse` bloğu; `run()` imzasini da guncelle |
| **Yeni test veri seti** | `make_data.py` sonuna fonksiyon | `data/*.csv` oldugu icin `report.py` otomatik alir |
| **REPORT.md sutunlarini degistirmek** | `report.py:36-49` (veri) + `:79-95` (tablo) | Iki yeri birlikte guncelle |
| **Bagimlilik surumu yukseltmek** | `requirements.txt` | Sonra `pyright automl` ve `pytest tests/` calistir |

---

## Ek: Temizlenmesi Onerilen Noktalar

Isleyisi **etkilemeyen** ama duzeltilmesi gereken bulgular:

| # | Yer | Sorun | Oneri |
|---|---|---|---|
| 1 | ~~`orchestrator.py` `PIPELINE`~~ | ✅ **DUZELTILDI** — olu kod silindi | — |
| 2 | `agents/profiler.py:2` | `import numpy as np` **hic kullanilmiyor** | Sil |
| 3 | `agents/modeler.py:23` | `_candidate_models(task_type, n_rows)` — `n_rows` kullanilmiyor | Sil veya kullan |
| 4 | `orchestrator.py:35` | `_yeterli_mi(metric_name, ...)` — `metric_name` kullanilmiyor | Sil |
| 5 | `orchestrator.py:155-156` | `_f()` `for` dongusunun **icinde** tanimli, her kolonda yeniden olusuyor | Dongu disina al |
| 6 | `schemas.py:7-8` | Yorum var olmayan `steps.py`'den bahsediyor | Yorumu guncelle |
| 7 | `schemas.py:72` | Pydantic modelinde `dataclasses.field()` — calisiyor ama stil tutarsizligi | `= []` yap |
| 8 | `agents/__init__.py` | 17 isim disa aciyor ama **kimse import etmiyor** | Birak veya sadelestir |
| 9 | `show_runs.py` | `if __name__ == "__main__"` yok — import edilirse hemen calisir | Guard ekle |
| 10 | `tests/` | LLM dogrulama katmaninin (`_kolonlar_gecerli`, `_secim_gecerli`, `_temizle`) **otomatik testi yok** | Test ekle — en onemli eksik |
| 11 | `runs/` | 50 klasor birikmis; `benzer_runlar` her planner cagrisinda hepsini okuyor | Periyodik temizlik |
