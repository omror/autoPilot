# autoPilot — Sunum ve Savunma Dokumani

---

## 1. Projenin Tek Cumlelik Iddiasi

> **Herhangi bir CSV dosyasini, kolon adlarini veya alanini onceden
> bilmeden alip; kolon tiplerini, hedef kolonu ve gorev tipini kendi
> tespit eden, preprocessing planini otomatik ureten, birden fazla
> modeli karsilastirip en iyisini secen ve sonuc yetersizse farkli
> stratejilerle kendini tekrar deneyen bir AutoML boru hatti.**

Iddianin can alici kismi **domain-bagimsizlik**: kod tabaninda hicbir
yerde kolon adi, veri seti adi veya alan bilgisi sabit kodlanmis
degildir. Ayni komut; cicek olcumleri (iris), kanser teshisi (cancer),
diyabet ilerlemesi (diabetes), musteri satin alma (sample), emlak
fiyati (sample_reg) ve kasitli olarak bozulmus karisik veri (messy)
uzerinde ayar degistirmeden calisir.

**Kanit:** `REPORT.md` — tek bir `python3 report.py` komutu, 7 veri
setini bastan sona isler; her biri icin gorev tipi, atilacak kolonlar
ve model secimi otomatik belirlenir.

---

## 2. Mimari

### 2.1 Genel Akis

Sistem 8 ajandan olusur. Her ajan `RunState` adli tek bir "canta"
nesnesini alir, kendi alanini doldurur ve geri dondurur.

```
                    ┌─────────────── RunState ───────────────┐
                    │  df, X_train, plan, model, result ...  │
                    └────────────────────────────────────────┘
                              ▲ her ajan okur/yazar

  HAZIRLIK (bir kez calisir)
  ┌─────────┐   ┌──────────┐   ┌──────────┐
  │ loader  │──▶│ profiler │──▶│ splitter │
  └─────────┘   └──────────┘   └──────────┘
   CSV oku      tip + task     train/test ayir
                tespiti        (sizinti onlemi)
                                     │
        ┌────────────────────────────┘
        ▼
  DENEME (her iterasyonda tekrar calisir)
  ┌─────────┐   ┌──────────────┐   ┌─────────┐   ┌───────────┐
  │ planner │──▶│ preprocessor │──▶│ modeler │──▶│ evaluator │
  └─────────┘   └──────────────┘   └─────────┘   └───────────┘
   plan uret     Pipeline kur       CV ile        test skoru +
   (+LLM)        fit SADECE train   model sec     importance
        ▲                                              │
        │      skor esigin altindaysa                  │
        └──────── strateji degistir, tekrar ───────────┘
                                                       ▼
                                                 ┌──────────┐
                                                 │ recorder │
                                                 └──────────┘
                                                  runs/ altina kaydet
```

### 2.2 Dosya Yapisi

```
automl/
  schemas.py          # Veri sozlesmeleri: ColumnProfile, DataProfile,
                      # PreprocessingPlan, ModelScore, RunResult, RunState
  orchestrator.py     # Ajanlari sirayla calistirir + self-improvement dongusu
  llm.py              # LLM cagrisi icin tek arayuz (opsiyonel katman)
  agents/
    base.py           # Agent soyut sinifi: state al, state dondur
    loader.py         # CSV oku
    profiler.py       # Tip tespiti, task tespiti, istatistik, korelasyon
    splitter.py       # Train/test ayrimi
    planner.py        # Preprocessing plani (+ LLM katmani + gecmis run'lar)
    preprocessor.py   # Plani sklearn Pipeline'a cevir, fit SADECE train
    modeler.py        # Aday modelleri CV ile karsilastir (+ LLM katmani)
    evaluator.py      # Test skorlari, Gini katsayisi, feature importance
    recorder.py       # runs/ altina kaydet
  memory/
    logger.py         # run.json + run.txt yaz
    store.py          # Gecmis run'lari oku, benzerlerini bul
tests/                # 30 test, 4 dosya + conftest
make_data.py          # 5 test veri seti uretir
report.py             # Tum veri setlerini calistirip REPORT.md uretir
```

Toplam ~2000 satir Python (testler dahil).

### 2.3 Neden Bu Tasarim?

**Neden her ajanin tek sorumlulugu var?**
Alternatif, tek bir `automl(df)` fonksiyonu yazmakti. Ayri ajanlara
bolmenin uc somut kazanci oldu:

1. **Test edilebilirlik.** `_detect_type` fonksiyonunu tek basina
   cagirip 9 farkli senaryoda test edebiliyorum. Monolitik bir
   fonksiyonda tip tespitini test etmek icin tum boru hattini
   calistirmak gerekirdi.
2. **Sizinti garantisi yapisal hale geliyor.** `splitter` ve
   `preprocessor` ayri ajanlar oldugu icin siralamalarini bir testle
   dogrulayabiliyorum (`test_split_preprocessten_once_calisiyor`).
   Tek fonksiyon olsaydi, sizinti ancak kodu gozle okuyarak
   yakalanabilirdi.
3. **Self-improvement dongusu bunu zorunlu kildi** (asagida).

**Neden `RunState` tek baglanti noktasi?**
Ajanlar birbirini **tanimaz**. `modeler`, `planner`'i import etmez;
sadece `state.plan`'i okur. Bu sayede:
- Bir ajanin ic mantigi degistiginde digerleri etkilenmez.
- Ajan sirasi degistirilebilir veya bir ajan atlanabilir.
- Testlerde sahte bir `RunState` kurup tek ajani calistirmak mumkun
  (`test_no_leakage.py` tam olarak bunu yapiyor: elle `X_train`,
  `X_test` ve `plan` kurup sadece `preprocess`'i cagiriyor).

**Self-improvement dongusu bu tasarimi neden gerektirdi?**
Dongu, boru hattini **ikiye bolmeyi** zorunlu kildi:

| Grup | Ajanlar | Kac kez calisir | Neden |
|---|---|---|---|
| `HAZIRLIK` | loader, profiler, splitter | **1 kez** | Veri ayni; tekrar okumak bos is. Daha onemlisi: split tekrarlanirsa her iterasyonda **farkli bir test seti** olusur ve iterasyonlarin skorlari karsilastirilamaz hale gelir. |
| `DENEME` | planner, preprocessor, modeler, evaluator | **her iterasyonda** | Strateji degisince plan, pipeline, model ve skor yeniden uretilmeli. |

Bu ayrim `orchestrator.py:26-30`'da acikca kodlanmistir. Monolitik bir
tasarimda "sadece sundan itibaren tekrar calistir" demek mumkun olmazdi.

---

## 3. Her Bilesen Ne Yapiyor

### 3.1 `loader` — Veri Okuma

| | |
|---|---|
| **Girdi** | `state.data_path` |
| **Cikti** | `state.df` |
| **Karar** | Yok — sadece `pd.read_csv()` |

Bilinckli olarak minimal tutuldu. **Onemli yan etki:** tarih ayristirmasi
yapilmaz (`parse_dates` kullanilmaz), bu yuzden CSV'deki tarih kolonlari
string olarak okunur. Sonucu Bolum 15'te sinir olarak isliyorum.

### 3.2 `profiler` — Veriyi Tanima

| | |
|---|---|
| **Girdi** | `state.df`, `state.target` (opsiyonel) |
| **Cikti** | `state.profile` (`DataProfile`) |
| **Kararlar** | Her kolonun tipi, hedef kolon, gorev tipi |

Uc is yapar:

1. **Her kolon icin tip tespiti** (`_detect_type`, Bolum 4.1).
2. **Tipe gore istatistik:** sayisal kolonlarda ortalama/std/min/q25/
   medyan/q75/max/carpiklik (`_numeric_stats`); kategorik kolonlarda
   en sik deger ve orani (`_categorical_stats`).
3. **Hedef ve gorev tespiti** (Bolum 4.2) + **korelasyon analizi**
   (`_high_correlations`, |r| >= 0.85, en fazla 15 cift).

### 3.3 `splitter` — Train/Test Ayrimi

| | |
|---|---|
| **Girdi** | `state.df`, `state.profile` |
| **Cikti** | `X_train`, `X_test`, `y_train`, `y_test` |
| **Karar** | Stratify uygulanacak mi |

`test_size=0.2`, `random_state=42`. Siniflandirmada **en az kalabalik
sinifta 2+ ornek varsa** stratify uygulanir — az ornekli sinifin test
setinden tamamen kaybolmasini onler. Stratify `ValueError` verirse
(cok kucuk veri) stratify'siz tekrar denenir; sistem cokmez.

### 3.4 `planner` — Preprocessing Plani

| | |
|---|---|
| **Girdi** | `state.profile`, `state.strateji` |
| **Cikti** | `state.plan` (`PreprocessingPlan`) |
| **Kararlar** | Hangi kolon atilir/kullanilir, imputation, PCA |

Kural-tabanli cekirdek (Bolum 4.3) + strateji katmani (Bolum 10) +
LLM katmani (Bolum 9) + gecmis run baglami (Bolum 11).

### 3.5 `preprocessor` — Plani Uygulama

| | |
|---|---|
| **Girdi** | `state.plan`, `X_train`, `X_test` |
| **Cikti** | `X_train_t`, `X_test_t`, `state.preprocessor` |
| **Karar** | Pipeline yapisi (PCA var mi) |

Sayisal dal: `SimpleImputer` → `StandardScaler`.
Kategorik dal: `SimpleImputer` → `OneHotEncoder(handle_unknown="ignore")`.
`ColumnTransformer` ile birlestirilir; `remainder="drop"` sayesinde
plana girmeyen kolonlar otomatik dusurulur. PCA aciksa `Pipeline`'a
ek adim olarak eklenir.

`handle_unknown="ignore"` kritik: test setinde egitimde gorulmemis bir
kategori cikarsa sistem cokmez, o kategori sifir vektorune donusur.

### 3.6 `modeler` — Model Secimi

| | |
|---|---|
| **Girdi** | `X_train_t`, `y_train`, `profile.task_type` |
| **Cikti** | `state.model`, `state.best_name`, `state.candidates` |
| **Kararlar** | Aday havuz, CV fold sayisi, kazanan model |

Bolum 7'de detayli.

### 3.7 `evaluator` — Degerlendirme

| | |
|---|---|
| **Girdi** | `state.model`, `X_test_t`, `y_test` |
| **Cikti** | `state.result` (`RunResult`) |
| **Kararlar** | Hangi metrikler, hangi importance |

Regresyon: r2, rmse, mae. Siniflandirma: accuracy, f1_weighted, +
ikili ise Gini katsayisi. Permutation importance her zaman; Gini
importance sadece agac modellerinde (Bolum 8).

### 3.8 `recorder` — Kalici Kayit

| | |
|---|---|
| **Girdi** | Tum `state` |
| **Cikti** | `runs/<timestamp>/run.json` + `run.txt` |
| **Karar** | Yok |

Bolum 11'de detayli.

---

## 4. Domain-Bagimsizlik Nasil Saglandi

**Bu projenin can alici noktasi.** Uc mekanizma birlikte calisir.

### 4.1 Otomatik Tip Tespiti

`profiler.py:14-42`. dtype'a **korukorune guvenmez**, veriye bakar.
Sirayla:

```
1. dtype gercekten datetime mi?          → datetime
2. dtype string/object mi?
     essiz > 50 VE essiz/satir > 0.5     → text
     degilse                             → categorical
3. dtype bool mu?                        → categorical
4. dtype sayisal mi?
     KURAL 1: ondalik deger var mi?      → numeric
     KURAL 2: essiz <= esik              → categorical
              essiz >  esik              → numeric
5. hicbiri                               → categorical
```

#### Kural 1 — Ondalik testi

```python
if not clean.mod(1).eq(0).all():
    return "numeric"
```

Bir kolonda tek bir ondalikli deger varsa o kolon kesin sayisaldir.
Kimse "0.5 sinif" ya da "3.7 kategori" kodlamaz. Bu kural ucuz
(tek gecis) ve **yanlis pozitif uretmez**, bu yuzden once calisir.

#### Kural 2 — Oransal esik

```python
esik = max(2, min(20, int(n_rows * 0.05)))
return "categorical" if clean.nunique() <= esik else "numeric"
```

**Neden sabit sayi degil, oran?**
Tamsayi kolonlarda `[0,1,2]` bir sinif kodu mu yoksa gercekten sayisal
bir olcum mu, sadece essiz deger sayisina bakarak anlasilamaz —
**veri boyutuna gore anlami degisir.**

- 1000 satirda sadece 8 essiz deger varsa: bu kolon buyuk olasilikla
  bir **kategori kodu**. 992 satir tekrar ediyor demektir.
- 40 satirda 8 essiz deger varsa: bu kolon buyuk olasilikla **sayisal**
  bir olcum. Kucuk orneklemde her degerin birkac kez tekrarlamasi
  normaldir.

Sabit bir esik (orn. "10'dan az essiz → kategorik") kucuk veri
setlerinde neredeyse **her kolonu kategorik** yapardi ve tum sayisal
bilgi one-hot encoding'e giderdi.

**Alt ve ust sinirlar ne ise yariyor?**

| Sinir | Deger | Ne oluyor olmasaydi |
|---|---|---|
| **Alt sinir** `max(2, ...)` | 2 | 20 satirlik veride `20*0.05 = 1` cikardi. Esik 1 olunca ikili (0/1) bir hedef kolonu bile "2 > 1" diye **sayisal** sayilir, ikili siniflandirma yanlislikla regresyona donerdi. |
| **Ust sinir** `min(20, ...)` | 20 | 100.000 satirlik veride `100000*0.05 = 5000` cikardi. 5000 essiz degerli bir kolon kategorik sayilir, one-hot encoding 5000 kolon uretir ve bellek patlardi. |

**Testle kanitlanmis:** `test_esik_veri_boyutuna_gore_degisir` — 8 essiz
degerli **ayni** kolon, `n_rows=1000` iken `categorical`, `n_rows=40`
iken `numeric` donuyor. Sabit esik kullanilsaydi bu test gecmezdi.

#### Metin kolonu ayrimi

`essiz > 50 AND essiz/satir > 0.5` — iki kosul birlikte aranir. Tek
basina "essiz > 50" yetersizdir: 10.000 satirlik veride 60 sehir adi
olan bir kolon gecerli bir kategoriktir. Ikinci kosul "her satir
neredeyse benzersiz" demektir; bu bir **serbest metin** imzasidir.
`messy.csv`'deki `aciklama` kolonu (400 satir, 400 essiz) bu yolla
yakalanip atiliyor.

### 4.2 Otomatik Task Tespiti

`profiler.py:121-141`.

```
hedef verilmemis mi?     → son kolon hedef kabul edilir
hedef hala yok mu?       → task_type = "clustering"
hedef veride yok mu?     → ValueError (mevcut kolonlari listeler)
hedefin tipi numeric mi? → "regression"
degilse                  → "classification" + sinif dagilimi hesaplanir
```

Kritik nokta: gorev tipi **hedef kolonun tespit edilen tipinden**
turetilir. Yani Bolum 4.1'deki tip tespiti dogru calismazsa task
tespiti de yanlis olur — ikisi zincirlidir. `sample.csv`'de
`satin_aldi` kolonu 0/1 degerli, 6 satirda 2 essiz; esik
`max(2, min(20, 0)) = 2` ve `2 <= 2` oldugu icin kategorik →
classification. Alt sinir olmasaydi bu ornek regresyona duserdi.

Olmayan hedef verilirse hata mesaji **mevcut kolonlari da listeler** —
kullanici yazim hatasini aninda gorur. `test_olmayan_hedef_anlamli_hata`
bunu dogruluyor.

### 4.3 Profile-Gudumlu Preprocessing Plani

**`planner` hicbir kolon adi bilmez.** Kodda tek bir kolon adi
gecmez; sadece `profile`'daki **ozelliklere** bakar.

Drop kurallari (sirayla, `planner.py:18-47`):

| # | Kural | Gerekce |
|---|---|---|
| 1 | Hedef kolonun kendisi | Ozellik olamaz — sizinti olur |
| 2 | `null_ratio > 0.5` | Yarisindan fazlasi bos; imputation gercek bilgi degil **uydurma** uretir. Model gurultuyle egitilir. |
| 3 | `inferred_type in ("text","datetime")` | Sistemde NLP ve tarih ozellik cikarimi yok. Ham haliyle one-hot'a sokulursa her satir kendi kolonu olur. |
| 4 | `categorical AND n_unique > 50` | One-hot patlamasi: 60 essiz kategori 60 yeni kolon demek. Boyut laneti + asiri ogrenme riski. |

Kalanlar tipine gore `numeric_cols` veya `categorical_cols`'a gider.

**PCA karari:** `use_pca = len(numeric_cols) > 10 and n_rows > 50`.
Iki kosul birlikte: cok kolon **ve** yeterli satir. `n_components =
min(10, len(numeric_cols))`.

**`messy.csv` uzerinde bu kurallarin canli sonucu:**

```
urun_kodu    : 60 essiz kategori, atildi      ← kural 4
kayit_tarihi : text tipi, atildi              ← kural 3
aciklama     : text tipi, atildi              ← kural 3
bos_kolon    : %100 bos, atildi               ← kural 2
```

Kalan: `olcum_a`, `olcum_b`, `olcum_c` (sayisal) + `seviye` (kategorik).
**%15 eksik degerli `olcum_a` atilmadi** — imputation ile kullanildi;
kural 2'nin esigi %50, dogru davranis bu.

---

## 5. Istatistik ve Veri Onizleme

### 5.1 Betimsel Istatistikler

Her sayisal kolon icin: ortalama, std, min, q25, medyan, q75, max,
**carpiklik (skew)**. Carpiklik bilincli bir tercih: dagilimin simetrik
olup olmadigini gosterir, aykiri deger ve donusum ihtiyacina isaret eder.

Ornek cikti (`diabetes.csv`):

```
  kolon                         ort        std        min     medyan        max  carpiklik
  s6                           0.00       0.05      -0.14      -0.00       0.14       0.21
  ilerleme                   152.13      77.09      25.00     140.50     346.00       0.44
```

Tamamen bos kolonlarda istatistik uretilemez; tablo `-` yazar
(bu koruma `messy.csv` gelistirilirken bulunan bir cokmeyi duzeltti).

### 5.2 Kategorik Dagilimlar

Her kategorik kolon icin en sik deger, orani ve essiz sayi:

```
  teshis                 en sik: 1 (%62.7), 2 essiz
  seviye                 en sik: orta (%34.5), 3 essiz
```

Siniflandirmada ayrica tam sinif dagilimi (`class_balance`) hesaplanir
ve `splitter`'in stratify karari ile `modeler`'in fold karari bunu kullanir.

### 5.3 Korelasyon Analizi

`_high_correlations`: sayisal kolonlar arasi Pearson korelasyonu, hedef
haric. |r| >= 0.85 olan ciftler, mutlak degere gore sirali, ilk 15.

`cancer.csv` cikti:

```
  mean radius     <-> mean perimeter    r=+0.998
  worst radius    <-> worst perimeter   r=+0.994
  mean radius     <-> mean area         r=+0.987
  mean perimeter  <-> mean area         r=+0.987
  worst radius    <-> worst area        r=+0.984
  ...
```

**Bu korelasyonlar PCA kararini nasil gerekcelendiriyor?**

Once **onemli bir durustluk notu**: kodda korelasyonlar PCA'yi
**tetiklemiyor**. Tetikleyici sadece kolon sayisi
(`len(numeric_cols) > 10 and n_rows > 50`). Korelasyonlar planner'in
kural-tabanli PCA kararina girmiyor; sadece (a) ekrana basiliyor ve
(b) LLM'e baglam olarak veriliyor.

Korelasyonlarin rolu **gerekcelendirme**, tetikleme degil:

- `mean radius` ile `mean perimeter` arasinda r=0.998 var. Bir dairenin
  cevresi yaricapinin 2πr katidir — bu iki kolon **ayni bilgiyi**
  tasiyor. `mean area` da (πr²) ayni ailede.
- 30 sayisal kolonun buyuk kismi bu sekilde birbirinin turevi. Yani
  **gercek boyut 30'dan cok daha kucuk.**
- PCA tam olarak bunu somurur: korele degiskenleri ortogonal
  bilesenlere cevirir. 30 kolon → 10 bilesen.
- **Sonuc bunu dogruluyor:** cancer'da PCA acikken f1 = 0.9737. Bilgi
  kaybi olsaydi skor duserdi; dusmedi.

Yani korelasyon tablosu, "kolon sayisi > 10" gibi kaba bir kuralin bu
veri setinde **neden dogru karari verdigini** gosteren kanittir.
Savunmada "korelasyonlar PCA'yi tetikliyor" dersem yanlis olur;
"korelasyonlar PCA'nin bu veride dogru secim oldugunu kanitliyor"
dogrudur.

---

## 6. Veri Sizintisi Onlemi

### 6.1 Neden Onemli — Somut Ornek

`StandardScaler` bir kolonu `(x - ortalama) / std` ile olcekler.
Ortalama ve std **fit** sirasinda ogrenilir.

Diyelim ki bir olcum kolonunda egitim verisi 0-100 arasinda, ama test
setinde 1.000.000 degerinde uc ornekler var. Eger scaler **tum veride**
fit edilirse:

- Ogrenilen ortalama 49.5 degil, milyonlara kayar.
- Egitim verisi bu carpik olcege gore donusur.
- Model, test setinin dagilimi hakkinda **onceden bilgi** edinmis olur.
- Rapor edilen test skoru gercekte olacagindan **yuksek** cikar.
- Model uretime alindiginda beklenenden kotu calisir — ve kimse
  neden oldugunu anlamaz.

Bu, ML projelerinde en sik yapilan ve en zor fark edilen hatadir.
Cunku **hicbir hata mesaji vermez**; sadece skoru sisirir.

### 6.2 Iki Katmanli Yapisal Onlem

**Katman 1 — Siralama.** `splitter`, `preprocessor`'dan **once**
gelir (`orchestrator.py:26-29`). Olcekleme ve imputation
hesaplanirken test seti henuz ayrilmis ve dokunulmamistir.
Dahasi `splitter` `HAZIRLIK` grubunda, `preprocessor` `DENEME`
grubunda — dongu ne kadar donerse donsun split hep once ve hep aynidir.

**Katman 2 — fit/transform ayrimi.** `preprocessor.py:49-51`:

```python
# KRITIK: fit sadece train'de, test'e sadece transform
state.X_train_t = preprocessor.fit_transform(state.X_train)
state.X_test_t  = preprocessor.transform(state.X_test)
```

`fit_transform` sadece `X_train`'e, `transform` `X_test`'e uygulanir.
Bu tek satirlik fark, tum garantinin dayandigi noktadir.

### 6.3 `tests/test_no_leakage.py` Bunu Nasil Test Ediyor

Bu dosya projenin en kritik testidir. 5 test icerir:

**1. `test_scaler_test_setinden_etkilenmiyor`** — Bolum 6.1'deki
senaryoyu birebir kurar:

```python
X_train = pd.DataFrame({"olcum": np.arange(100, dtype=float)})  # 0-99
X_test  = pd.DataFrame({"olcum": [1_000_000.0] * 25})           # uc degerler
...
assert scaler.mean_[0] == 49.5   # train'in ortalamasi, degismemis
assert st.X_test_t.min() > 1000  # uc degerler devasa z-skoru olmus
```

Fit test setini de gorseydi ortalama 49.5'te kalmaz, milyonlara kayardi.
Ikinci assert de onemli: test verisi **transform edilmis ama fit'e
katilmamis** olmali; devasa z-skoru bunun kanitidir.

**2. `test_test_setine_sadece_transform_uygulaniyor`** — Donusturulmus
train ortalamasi ~0 (merkezlenmis), test ortalamasi > 5 (kaymis kalmis).
Ikisi de merkezlenseydi test seti fit'e karismis olurdu.

**3. `test_bos_deger_imputation_da_traindan_ogreniliyor`** — Sizinti
sadece scaler'da degil, **imputer'da** da olabilir. Train medyani 3.0;
test'teki 999.999 bu degeri etkilememeli.

**4. `test_split_preprocessten_once_calisiyor`** — Yapisal test:
`HAZIRLIK + DENEME` listesinde `splitter` indeksi `preprocessor`
indeksinden kucuk olmali. Biri siralamayi degistirirse test kirmizi olur.

**5. `test_uctan_uca_scaler_sadece_traini_gormus`** — Gercek veride
(`iris.csv`) uctan uca: scaler'in ortalamasi `X_train`'in ortalamasina
esit, **ve** tum verinin ortalamasina esit **degil**. Ikinci assert
kritik: ikisi tesadufen esit olsaydi test hicbir sey kanitlamazdi.

---

## 7. Model Secimi ve Degerlendirme

### 7.1 Aday Havuz

| Task | Modeller |
|---|---|
| **Classification** | `DummyClassifier(most_frequent)`, `LogisticRegression(max_iter=1000)`, `RandomForestClassifier(n_estimators=100)`, `GradientBoostingClassifier` |
| **Regression** | `DummyRegressor(mean)`, `LinearRegression`, `RandomForestRegressor(n_estimators=100)`, `GradientBoostingRegressor` |

Tumu `random_state=42` — tekrarlanabilirlik icin.

Havuz bilincli olarak **cesitlilige** gore secildi: bir dogrusal model,
bir bagging ensemble, bir boosting ensemble. Bu uclu, farkli veri
yapilarini kapsar — dogrusal ayrilabilir veride LogisticRegression,
dogrusal olmayan etkilesimlerde agac modelleri kazanir. `nonlinear.csv`
tam olarak bunu gostermek icin uretildi (RandomForest kazandi).

### 7.2 Baseline Neden Var?

`DummyClassifier(strategy="most_frequent")` her zaman en kalabalik
sinifi tahmin eder; `DummyRegressor(strategy="mean")` her zaman
ortalamayi.

**Baseline olmadan bir skor anlamsizdir.** %95 accuracy iyi mi? Veri
%95 tek siniftan olusuyorsa **hayir** — hicbir sey ogrenmemis bir model
de %95 alir. Baseline, "modelin ogrendigi seyin degeri" icin sifir
noktasidir.

Somut kanit (`cancer.csv`):

```
   Baseline             cv=0.482
 * LogisticRegression   cv=0.980
```

Baseline 0.482, kazanan 0.980. Aradaki fark **gercekten ogrenilen
bilgidir**. `nonlinear.csv`'de baseline 0.332, kazanan 0.753 — daha
zor bir problem oldugu buradan okunur.

### 7.3 CV Fold Sayisi Nasil Ayarlaniyor

`modeler.py:51-54`:

```python
n_splits = max(2, min(5, len(y) // 2))
if p.task_type == "classification":
    n_splits = max(2, min(n_splits, int(y.value_counts().min())))
```

Iki asamali:
1. **Genel sinir:** en fazla 5 fold, ama her fold'da en az 2 ornek
   kalacak sekilde (`len(y) // 2`), en az 2 fold.
2. **Siniflandirmada ek sinir:** fold sayisi **en az kalabalik sinifin
   ornek sayisini** gecemez. Gecerse bazi fold'larda o siniftan hic
   ornek olmaz ve `cross_val_score` patlar.

Sabit `cv=5` kullanilsaydi `sample.csv` (6 satir) veya `sample_reg.csv`
(8 satir) gibi kucuk veri setlerinde sistem cokerdi. **REPORT.md'de
6 satirlik veri setinin de calismis olmasi bu mantigin kanitidir.**

Ayrica her model `try/except` icinde denenir; biri patlarsa atlanir ve
digerleri devam eder. Hicbiri calismazsa acik hata verilir.

### 7.4 Metrik Secimi

| Task | Ana metrik | Neden |
|---|---|---|
| Classification | `f1_weighted` | Accuracy dengesiz veride yanilticidir. Weighted F1 her sinifin precision/recall dengesini sinif buyuklugune gore agirliklandirir. |
| Regression | `r2` | Olcek-bagimsiz; farkli veri setleri arasinda karsilastirilabilir. RMSE/MAE hedefin birimine bagli oldugu icin ana metrik olamaz (ama raporlanir). |

Model secimi CV skoruna gore yapilir (`X_train` uzerinde), nihai rapor
**dokunulmamis test setine** gore verilir. Bu ayrim onemli: CV skoru
model secmek icin, test skoru durust raporlama icin.

---

## 8. PCA ve Gini

### 8.1 PCA — Kosullu Devreye Girme

```python
use_pca = len(numeric_cols) > 10 and p.n_rows > 50
n_components = min(10, len(numeric_cols))
```

**Neden iki kosul?**
- `> 10 kolon`: az kolonda PCA'nin kazanci yok, sadece yorumlanabilirlik
  kaybi olur.
- `> 50 satir`: PCA kovaryans matrisi tahmin eder. Cok az satirda bu
  tahmin gurultulu olur ve PCA veriyi iyilestirmek yerine bozar.

REPORT.md'de PCA'nin dagilimi: `cancer` (30 sayisal kolon) → 10 bilesen,
`nonlinear` (12 kolon) → 10 bilesen. `iris` (4 kolon), `diabetes`
(10 kolon — tam sinirda, `10 > 10` yanlis), `messy` (3 kolon) → PCA yok.

`diabetes` ilginc bir sinir vakasi: tam 10 sayisal kolon var, kural
`> 10` oldugu icin PCA acilmiyor. Bu keyfi degil — 10 kolon zaten
yonetilebilir bir boyut.

### 8.2 Gini — Uc Farkli Anlam

Hocamin acik sorusu buydu: *"gini indeksine bakilmasi gerekiyor mu?"*
Cevap vermeden once **terimin uc ayri anlamini ayirmak** gerekiyor,
cunku literaturde ayni isimle anilirlar ve karistirilirlar.

| # | Ad | Ne olcer | Nerede kullanilir |
|---|---|---|---|
| **1** | **Gini impurity** (safsizlik) | Bir dugumdeki sinif karisikligi: `1 - Σpᵢ²` | Karar agaci **bolunme kriteri** |
| **2** | **Gini importance** (MDI) | Bir ozelligin agac boyunca sagladigi toplam safsizlik azalmasi | **Ozellik onemi** |
| **3** | **Gini katsayisi** | Modelin ayirt etme gucu: `2·AUC - 1` | **Model performans metrigi** |

Birbirlerinden turemis olsalar da **farkli seyleri olcerler**. 1 ve 2
model *icinde*, 3 model *disinda* yasar.

#### Anlam 1 — Gini impurity (dolayli olarak var)

`RandomForestClassifier`'in sklearn'deki varsayilan bolunme kriteri
`criterion="gini"`'dir (dogruladim). Yani sistemde RandomForest
kazandiginda, agaclarin her dugumu Gini impurity ile bolunmustur.
Bu **acikca kodlanmadi**, sklearn varsayilanidir.

Not: `GradientBoostingClassifier`'da bu parametre artik `deprecated`;
`RandomForestRegressor` ise `squared_error` kullanir. Yani "tum agac
modellerim Gini kullaniyor" demek yanlis olur.

#### Anlam 2 — Gini importance (acikca eklendi)

`evaluator._gini_importance` — modelin `feature_importances_`
ozelligini okur, buyukten kucuge siralar, ilk 10'u dondurur.

Savunmalar: `feature_importances_` yoksa bos dict; deger sayisi
ozellik sayisiyla uyusmuyorsa bos dict; hata olursa bos dict.
Yani **sadece agac modellerinde** dolar.

#### Anlam 3 — Gini katsayisi (acikca eklendi)

`evaluator._gini_coefficient` — `2 * roc_auc_score - 1`.

0 = rastgele tahmin, 1 = mukemmel ayirim. Kredi skorlamada AUC yerine
yaygin olarak bu kullanilir. Sadece **ikili siniflandirmada** anlamli
oldugu icin sinif sayisi 2 degilse `None` doner; `predict_proba`
olmayan modellerde de `None` doner.

### 8.3 Ana Metrik Neden Permutation Importance?

Sistemde **her iki importance da hesaplanir**, ama ana rapor
permutation importance uzerinden verilir. Gerekce:

| Kriter | Permutation importance | Gini importance (MDI) |
|---|---|---|
| **Model kapsami** | Her modelde calisir (model-agnostik) | Sadece agac modellerinde |
| **Olculen sey** | **Test** setinde performans dususu | **Egitim** verisindeki safsizlik azalmasi |
| **Bilinen yanlilik** | Korele ozelliklerde onemi paylastirir | **Yuksek kardinaliteli ozellikleri sisirir** |
| **Sizinti riski** | Yok — test setinde olculur | Egitim verisine bakar, asiri ogrenmeyi yansitabilir |
| **Maliyet** | Pahali (n_repeats × yeniden tahmin) | Bedava (egitimin yan urunu) |

Iki belirleyici sebep:

1. **Model-agnostiklik.** Sistem hangi modelin kazanacagini onceden
   bilmiyor. Ana metrik Gini importance olsaydi, LogisticRegression
   veya LinearRegression kazandiginda **hicbir importance
   raporlanamazdi.** REPORT.md'deki 7 veri setinin **4'unde** dogrusal
   model kazaniyor — yani vakalarin cogunlugunda tablo bos kalirdi.

2. **MDI'nin yuksek kardinalite yanliligi.** Gini importance, cok
   essiz degerli ozellikleri sistematik olarak abartir; cunku boyle
   ozellikler daha cok bolunme firsati bulur. Permutation importance
   test setinde gercek performans etkisini olctugu icin bu yanliliktan
   muaftir.

### 8.4 Somut Kanit — Iki Veri Setinin Karsilastirmasi

Bu karsilastirma, permutation'in neden ana metrik secildiginin
**dogrudan kanitidir**:

**`cancer.csv` — LogisticRegression kazandi:**

```
--- TEST SONUCU (LogisticRegression) ---
  accuracy     : 0.9737
  f1_weighted  : 0.9737
  gini         : 0.9907          ← Gini KATSAYISI var (anlam 3)

--- FEATURE IMPORTANCE ---        ← permutation, dolu
  pca0                      +0.4263
  pca1                      +0.1105
  ...
  not: secilen model agac-bazli degil, Gini importance hesaplanamaz
                                  ← Gini IMPORTANCE yok (anlam 2)
```

**`nonlinear.csv` — RandomForest kazandi:**

```
--- TEST SONUCU (RandomForest) ---
  accuracy     : 0.8438
  f1_weighted  : 0.8433
  gini         : 0.8177          ← Gini KATSAYISI var

--- FEATURE IMPORTANCE ---        ← permutation, dolu
  pca2                      +0.1512
  ...
--- GINI IMPORTANCE (agac-bazli) ---   ← Gini IMPORTANCE de var
  pca2                      0.1761
  pca0                      0.1602
  ...
  not: secilen model agac-bazli oldugu icin hesaplandi
```

**Okunacak sonuc:** Gini **katsayisi** her ikili siniflandirmada
uretilebiliyor (model bagimsiz, `predict_proba` yeterli). Gini
**importance** ise sadece agac kazandiginda uretilebiliyor. Ana metrik
Gini importance olsaydi cancer icin hicbir ozellik onemi
raporlanamazdi. **Permutation her iki durumda da dolu.**

Ayrica nonlinear'da iki siralamayi karsilastirmak mumkun: permutation
`pca2 > pca1 > pca0`, Gini `pca2 > pca0 > pca1`. Ust sira ayni ama
alt siralar farkli — iki yontemin farkli seyler olctugunun somut
gostergesi.

**Hocanin sorusuna cevap:** Evet, Gini'ye bakilmasi gerekiyordu —
ama hangisine? Sistemde **ucu de** ele alindi: impurity sklearn
varsayilaniyla dolayli, importance ve katsayi acikca kodlandi.
Importance ana metrik yapilmadi cunku model-bagimli.

---

## 9. LLM Katmani

### 9.1 Nerede Devreye Giriyor

Iki ajanda, ikisinde de **mevcut kural-tabanli mantigin uzerine**:

| Ajan | LLM'e sorulan | Dogrulanan sema |
|---|---|---|
| `planner` | "Bu profile bakarak preprocessing planini dondur" | `PreprocessingPlan` |
| `modeler` | "Bu havuzdan hangi modeller denenmeli?" | `ModelSecimi` |

Model: `claude-sonnet-4-5`. Anahtar `ANTHROPIC_API_KEY` ortam
degiskeninden okunur.

### 9.2 Ne Gonderiliyor

`planner` icin (`_profil_ozeti`):
- Satir/kolon sayisi, hedef kolon, gorev tipi
- Her kolonun adi, tespit edilen tipi, essiz sayisi, null orani
- Yuksek korelasyon ciftleri
- **Kural-tabanli planin ne onerdigi** (referans nokta olarak)
- Benzer gecmis run'lar ve skorlari (Bolum 11)
- Acik talimat: "Kolon adlarini AYNEN listeden kullan, uydurma"

`modeler` icin (`_model_ozeti`): profil ozeti, sinif dagilimi, ozellik
sayilari, PCA durumu ve **mevcut model havuzu**.

### 9.3 Cikti Nasil Valide Ediliyor — Dort Katman

`llm.py` ve ajanlarda toplam dort katman:

**Katman 1 — JSON parse.** Markdown kod bloklari (` ```json `)
temizlenir, `json.loads` denenir. Basarisizsa `None`.

**Katman 2 — Sema dogrulamasi.** `schema_class.model_validate(veri)`
ile pydantic dogrulamasi. Eksik/yanlis tipli alan varsa `None`.

**Katman 3 — Uydurma kontrolu (kritik).**

| Kontrol | Nerede | Ne yapiyor |
|---|---|---|
| **Uydurma kolon adi** | `_kolonlar_gecerli` | LLM'in onerdigi her kolon adi gercekten veride var mi? Yoksa plan tumden reddedilir. |
| **Hedef sizmasi** | `_kolonlar_gecerli` | Gecerli kume = *profildeki kolonlar eksi hedef*. LLM hedefi ozellik listesine koyarsa reddedilir — hem sizinti hem cokme onlenir. |
| **Bos ozellik listesi** | `_kolonlar_gecerli` | Hem numeric hem categorical bossa reddedilir (pipeline kurulamazdi). |
| **Uydurma model adi** | `_secim_gecerli` | LLM havuzda olmayan bir model onerirse (orn. "XGBoost") secim reddedilir. |
| **Bos model listesi** | `_secim_gecerli` | Reddedilir. |

**Katman 4 — Genel istisna yakalama.** `ask_json`'un tum govdesi tek
bir `try/except Exception` icinde. **Fonksiyon hicbir kosulda istisna
firlatmaz**; ya gecerli nesne ya `None` doner.

### 9.4 Fallback Mantigi — Hangi Durumlarda Kural-Tabanliya Duser

| Durum | Sonuc |
|---|---|
| `ANTHROPIC_API_KEY` tanimli degil | LLM hic cagrilmaz |
| `anthropic` paketi kurulu degil | LLM hic cagrilmaz |
| Ag hatasi / zaman asimi | `None` → kural-tabanli |
| Gecersiz anahtar (401) | `None` → kural-tabanli |
| Bozuk JSON | `None` → kural-tabanli |
| Sema dogrulamasi basarisiz | `None` → kural-tabanli |
| Uydurma kolon adi | Oneri reddedilir → kural-tabanli |
| Hedef kolonu ozellik listesinde | Oneri reddedilir → kural-tabanli |
| Uydurma model adi | Secim reddedilir → tum havuz denenir |

Her durumda plan notlarina `"kural-tabanli plan kullanildi"` yazilir —
hangi yolun kullanildigi ciktidan okunabilir.

**Tasarim ilkesi:** kural-tabanli mantik **her zaman once calisir** ve
guvenli tabani olusturur. LLM bir *ikinci gorustur*, tek karar verici
degil. Bu, sistemin LLM olmadan **tam islevsel** kalmasini saglar.

### 9.5 Gercek Test Sonuclari

**Birim testleri** (koruma mantigi):

```
markdown fence temizleme  : '```json\n{"a": 1}\n```' → '{"a": 1}'  ✓
is_available() anahtarsiz : False                                  ✓
is_available() anahtarli  : True                                   ✓
uydurma kolon 'uydurma_kolon' → reddedildi                         ✓
hedef sizmasi 'y'            → reddedildi                          ✓
uydurma model 'XGBoost'      → reddedildi                          ✓
bos model listesi            → reddedildi                          ✓
```

**Gecersiz anahtarla uctan uca test** (fallback'in asil kaniti):

```
$ export ANTHROPIC_API_KEY="sk-ant-gecersiz-test-anahtari"
$ python3 -m automl.orchestrator --data data/iris.csv

   ! LLM cagrisi basarisiz: AuthenticationError: Error code: 401 ...
   ! LLM cagrisi basarisiz: AuthenticationError: Error code: 401 ...
    kural-tabanli model havuzu kullanildi
    secilen model: LogisticRegression (cv=0.958)
  not: kural-tabanli plan kullanildi
  f1_weighted  : 0.9333
```

LLM yolu **gercekten devreye girdi** (planner + modeler, 2 cagri),
ikisi de 401 aldi, sistem cokmedi ve skor degismedi (0.9333).
Bu, "anahtar yok" durumundan daha guclu bir testtir: anahtar varken
cagri yapilir ve **hata yolu** calisir.

**Not:** Gercek bir API anahtariyla calistirma yapilmadi (ortamda
anahtar yok). Yani LLM'in *iyilestirici* etkisi olculmedi; sadece
*zarar vermedigi* ve *fallback'in calistigi* kanitlandi. Bu bir
sinirdir, Bolum 15'te tekrar geciyor.

---

## 10. Self-Improvement Dongusu

### 10.1 Nasil Calisiyor

```
HAZIRLIK bir kez calisir (loader → profiler → splitter)
  ↓
for i in 1..max_iterasyon:
    strateji = STRATEJILER[i-1]
    DENEME calisir (planner → preprocessor → modeler → evaluator)
    sonucu gecmis_denemeler'e kaydet
    en iyiyse anlik goruntusunu sakla
    esik gecildiyse → DUR
  ↓
EN IYI iterasyonu geri yukle (son iterasyonu degil)
  ↓
recorder ile kaydet
```

### 10.2 Esikler

```python
ESIKLER = {"classification": 0.80, "regression": 0.50}
```

Siniflandirmada `f1_weighted < 0.80`, regresyonda `r2 < 0.50` yetersiz
sayilir ve yeni strateji denenir.

### 10.3 Stratejiler

| # | Strateji | Ne degistiriyor |
|---|---|---|
| 1 | `varsayilan` | Mevcut kurallar (degisiklik yok) |
| 2 | `pca_ters` | PCA aciksa kapatir, kapaliysa acar (en az 2 sayisal kolon varsa) |
| 3 | `imputation_degis` | median→mean, most_frequent→constant |

### 10.4 En Iyi Iterasyon Nasil Seciliyor

`_goruntu_al` / `_goruntu_yukle` ile 10 alanin anlik goruntusu saklanir:
`plan`, `result`, `model`, `best_name`, `candidates`, `preprocessor`,
`X_train_t`, `X_test_t`, `strateji`, `iterasyon`.

Karsilastirma **kesin buyuktur** (`>`) ile yapilir — beraberlikte ilk
iterasyon korunur (daha basit plan tercih edilir).

### 10.5 Sonsuz Dongu Korumasi

Uc katman:
1. `for i in range(1, max_iterasyon + 1)` — sabit ust sinir.
2. Esik gecilince `break`.
3. Bir iterasyon istisna firlatirsa `try/except` yakalar, hatayi
   `gecmis_denemeler`'e yazar ve **sonraki iterasyona devam eder**.
   Tum iterasyonlar patlarsa acik `RuntimeError`.

### 10.6 DURUSTCE: Stratejiler Zayif

**Bu bolumun en zayif noktasi ve savunmada kendim soyleyecegim.**

`diabetes.csv`'de uc strateji de **ayni skoru** verdi:

```
 * 1. varsayilan         r2=0.4526  PCA=False imp=median
   2. pca_ters           r2=0.4526  PCA=True  imp=median
   3. imputation_degis   r2=0.4526  PCA=False imp=mean
```

Stratejiler **uygulandi** (tablodaki PCA/imp sutunlari degisiyor), ama
skoru degistirmedi. Sebebi matematiksel:

1. **`pca_ters` etkisiz:** diabetes'te 10 sayisal kolon var ve
   `n_components = min(10, 10) = 10`. Yani PCA, 10 boyutu 10 boyuta
   donusturuyor — bu **tam rankli ortogonal bir dondurme**, bilgi
   kaybi yok. `LinearRegression` tersinir dogrusal donusume
   **degismezdir** (invariant): `X` yerine `XA` verirseniz katsayilar
   `A⁻¹β` olur ve tahminler **birebir ayni** kalir. Dolayisiyla r2
   degismesi matematiksel olarak imkansiz.

2. **`imputation_degis` etkisiz:** diabetes'te **hic eksik deger yok**
   (dogruladim: 0). Imputer hicbir hucreyi doldurmuyor, median ile
   mean arasinda secim yapmanin sonuca etkisi yok.

Yani bu iki strateji bu veri seti icin **yapisal olarak etkisiz**.
Sistem 3 iterasyon harcadi ve hicbir sey kazanmadi.

### 10.7 AMA: Dongu Gercekten Ise Yariyor — Iki Kanit

**Kanit 1 — `sample.csv`: skor gercekten artti.**

```
   1. varsayilan   f1_weighted=0.3333  PCA=False(None)
 * 2. pca_ters     f1_weighted=1.0000  PCA=True(2)
```

Iterasyon 1 esigin (0.80) altinda kaldi → strateji degisti → PCA
acildi → skor 0.3333'ten 1.0000'e cikti → esik gecildi → durdu.

**Uyari (durustluk):** `sample.csv` 6 satir; test seti ~2 satir.
f1=1.0 demek "2/2 dogru" demek — **istatistiksel olarak anlamsiz**.
Bu ornek dongunun *mekanizmasinin calistigini* kanitlar, skorun
gercekten iyi oldugunu degil.

**Kanit 2 — `sample_reg.csv`: kotu iterasyon dogru sekilde atildi.**

```
 * 1. varsayilan         r2= 0.1837  PCA=False
   2. pca_ters           r2=-2.5829  PCA=True(2)   ← felaket
   3. imputation_degis   r2= 0.1837  PCA=False
```

Iterasyon 2 skoru **-2.58**'e cakti (baseline'dan kotu). Sistem son
iterasyonu degil **en iyisini** (iterasyon 1) rapor etti. "Son sonuc
degil en iyi sonuc" mantiginin somut kaniti budur.

---

## 11. Hafiza ve Loglama

### 11.1 `runs/` Yapisi

Her calistirma `runs/<YYYYMMDD_HHMMSS>/` altina iki dosya yazar.
Su an dizinde **48 run** birikmis durumda.

**`run.json`** — makine okunur, 9 anahtar:

| Anahtar | Icerik |
|---|---|
| `timestamp` | Calistirma zamani |
| `data_path` | Hangi veri seti |
| `sure_sn` | Toplam sure |
| `profile` | Tam `DataProfile` (7 alan; her kolonun tipi, istatistigi) |
| `plan` | Tam `PreprocessingPlan` (10 alan) |
| `result` | Tam `RunResult` (8 alan; metrikler, importance'lar) |
| `iterasyonlar` | **Her iterasyonun** stratejisi, skoru, PCA/imputation ayari |
| `en_iyi_iterasyon` | Hangi iterasyon raporlandi |
| `en_iyi_strateji` | Kazanan strateji |

**`run.txt`** — insan okunur ozet:

```
Zaman   : 20260903_133726
Veri    : data/nonlinear.csv
Sure    : 3.78 sn
Boyut   : 800 satir x 13 kolon
Hedef   : hedef  (classification)
Sayisal : 12 kolon
PCA     : True (10)
Model   : RandomForest
Skor    : f1_weighted = 0.8433
Iterasyon: 3 deneme, en iyisi 1 (varsayilan)
  1. varsayilan         f1_weighted=0.8433 model=RandomForest PCA=True imp=median
  2. pca_ters           f1_weighted=0.8122 model=RandomForest PCA=False imp=median
  3. imputation_degis   f1_weighted=0.8433 model=RandomForest PCA=True imp=mean
```

`show_runs.py` tum run'lari tek satirlik ozetlerle listeler.

### 11.2 Benzerlik Skoru — Uc Sinyal

`store._benzerlik(p1, p2)` iki veri profilini karsilastirir:

| Sinyal | Puan | Mantik |
|---|---|---|
| **Ayni task tipi** | 0.4 | Farkli task ise **0.0 doner** (kapi kosulu). Regresyon run'i siniflandirmaya ornek olamaz. |
| **Veri boyutu 2 kat icinde** | +0.3 | `max(r1,r2)/min(r1,r2) <= 2`. 150 satirlik veriden ogrenilen, 100.000 satirlik veriye uymaz. |
| **Sayisal kolon orani yakin** | +0.3 | `abs(oran1-oran2) < 0.25`. Tamamen sayisal bir veri ile tamamen kategorik bir veri farkli preprocessing ister. |

Toplam max 1.0. `benzer_runlar(profile, esik=0.6, limit=5)` — 0.6
esigini gecen en benzer 5 run dondurulur. Esik 0.6 demek: ayni task
(0.4) **artik yetmez**, en az bir yapisal sinyalin de tutmasi gerekir.

### 11.3 Self-Improvement'a Baglanti

`PlannerAgent.run` icinde `_gecmis_ozeti(p)` cagrilir:

- **LLM varsa:** gecmis run'lar prompt'a baglam olarak eklenir —
  *"Benzer veride daha once denenen planlar: benzerlik=1.00 |
  PCA=True(10) | imputation=median/most_frequent |
  model=LogisticRegression | f1_weighted=0.9737 ... Bu gecmis
  sonuclari dikkate al, daha iyi skor veren plana yaklas."*
- **LLM yoksa:** ayni bilgi plan notlarina yazilir (bilgi amacli).

Canli ornek (`cancer.csv`, LLM kapali):

```
  not: gecmis: benzerlik=1.00 | PCA=True(10) | imputation=median/most_frequent
       | model=LogisticRegression | f1_weighted=0.9737
```

**Durustluk notu:** LLM yokken bu bilgi **karari etkilemez**, sadece
raporlanir. Yani "gecmisten ogrenme" su an tam anlamiyla ancak LLM
acikken devreye girer. Kural-tabanli yolda gecmis sadece gorunurdur.

---

## 12. Testler

**30 test, 4 dosya**, `python3 -m pytest tests/ -v` ile **1.76 saniyede**
gecer. `conftest.py`'deki autouse fixture `ANTHROPIC_API_KEY`'i siler —
testler LLM'e cikmaz, **deterministik ve ucretsiz** kalir.

| Dosya | Test | Kapsanan kritik yol |
|---|---|---|
| `test_type_detection.py` | 9 | Tip tespitinin her dali |
| `test_task_detection.py` | 5 | Hedef ve gorev tespiti |
| `test_no_leakage.py` | **5** | **Veri sizintisi** |
| `test_pipeline.py` | 11 | Uctan uca zorlu veri |

### 12.1 `test_type_detection.py` (9 test)

Ondalikli→numeric, az kardinaliteli int→categorical, cok kardinaliteli
int→numeric, string→categorical, her satiri farkli string→text,
bool→categorical, gercek datetime→datetime, **esigin veri boyutuna gore
degismesi**, tamamen bos kolon cokmuyor.

En degerlisi `test_esik_veri_boyutuna_gore_degisir`: **ayni** Series
nesnesi, `n_rows=1000` iken categorical, `n_rows=40` iken numeric.
Bu test oransal esigin varlik sebebini kanitlar.

### 12.2 `test_task_detection.py` (5 test)

Sayisal hedef→regression, kategorik hedef→classification, hedef
verilmezse son kolon, olmayan hedef→anlamli `ValueError` (mesajda hem
eksik kolon adi hem mevcut kolonlar), profile olmadan plan→`RuntimeError`.

### 12.3 `test_no_leakage.py` (5 test) — EN KRITIK

Bolum 6.3'te detayli anlatildi. Ozet: scaler test setinden etkilenmiyor,
test'e sadece transform uygulaniyor, imputer da train'den ogreniyor,
split yapisal olarak once geliyor, gercek veride scaler ortalamasi
`X_train`'e esit ve tum veriye esit **degil**.

**Neden en kritik:** Veri sizintisi hicbir hata mesaji vermez; sadece
skoru sisirir. Bu testler olmadan "sizinti yok" demek bir **iddia**
olurdu; testlerle birlikte **kanit** oluyor.

### 12.4 `test_pipeline.py` (11 test)

`messy.csv` uzerinde uctan uca (module-scope fixture, bir kez calisir):
cokmuyor, task classification, `bos_kolon` atildi, `urun_kodu` atildi,
`aciklama` atildi, `kayit_tarihi` atildi, kullanilabilir kolonlar
korundu, %15 eksikli kolon atilmadi, sonuc uretildi, iterasyon kaydi
tutuldu, **donusturulmus matriste NaN kalmadi**.

Son test onemli: imputation gercekten calisti mi? `X_train_t` ve
`X_test_t` icinde tek bir NaN kalirsa model egitimi patlardi.

---

## 13. Sonuclar Tablosu

`REPORT.md`'den, `python3 report.py` ile uretilen gercek sayilar:

| veri seti | satir x kolon | task | atilan kolon | PCA | model | metrik | skor | iterasyon | sure |
|---|---|---|---|---|---|---|---|---|---|
| cancer.csv | 569 x 31 | classification | 0 | 10 | LogisticRegression | f1_weighted | **0.9737** | 1 | 0.7 sn |
| diabetes.csv | 442 x 11 | regression | 0 | - | LinearRegression | r2 | **0.4526** | 3 | 2.0 sn |
| iris.csv | 150 x 5 | classification | 0 | - | LogisticRegression | f1_weighted | **0.9333** | 1 | 0.6 sn |
| messy.csv | 400 x 9 | classification | **4** | - | GradientBoosting | f1_weighted | **0.8625** | 1 | 0.5 sn |
| nonlinear.csv | 800 x 13 | classification | 0 | 10 | RandomForest | f1_weighted | **0.8433** | 1 | 1.2 sn |
| sample.csv | 6 x 4 | classification | 0 | 2 | GradientBoosting | f1_weighted | **1.0000** | 2 | 0.3 sn |
| sample_reg.csv | 8 x 4 | regression | 0 | - | LinearRegression | r2 | **0.1837** | 3 | 0.5 sn |

### 13.1 Tablodan Okunacak Cikarimlar

**1. Her boyutta calisiyor — 6 satirdan 569 satira.**
`sample.csv` 6 satir, `nonlinear.csv` 800 satir. Arada ~133 kat fark
var ve ikisi de cokmeden isleniyor. Bu, CV fold sayisinin otomatik
ayarlanmasinin (Bolum 7.3) ve tip esiginin oransal olmasinin
(Bolum 4.1) dogrudan sonucudur. Sabit `cv=5` veya sabit tip esigi
kullanilsaydi kucuk veri setleri cokerdi.

**2. Her uc task tipi de dogru tespit ediliyor.**
5 classification, 2 regression — hicbiri elle belirtilmedi. Hedef kolon
adlari birbirinden tamamen farkli (`teshis`, `ilerleme`, `tur`, `hedef`,
`satin_aldi`, `fiyat`) ve sistem hicbirini bilmiyor.

**3. Farkli alanlar, ayni kod.**
Tibbi teshis (cancer), botanik (iris), tibbi ilerleme (diabetes),
perakende (sample: yas/maas/sehir), emlak (sample_reg: metrekare/
oda_sayisi/ilce), sentetik (nonlinear, messy). Turkce ve Ingilizce
kolon adlari karisik. **Tek satir kod degismedi.**

**4. `messy.csv` drop kararlari dogru.**
Tek "atilan kolon > 0" olan veri seti: 4 kolon atildi. Hangileri
oldugu Bolum 4.3'te; her biri farkli bir kural tarafindan yakalandi
(bos kolon, yuksek kardinalite, iki metin). Diger veri setlerinde
atilacak kolon yok ve sistem **gereksiz yere kolon atmiyor** — bu da
onemli, asiri agresif bir planner her veriden kolon atardi.

**5. Model secimi veriye gore degisiyor.**
LogisticRegression (cancer, iris), LinearRegression (diabetes,
sample_reg), RandomForest (nonlinear), GradientBoosting (messy,
sample). Tek bir model dayatilmiyor; CV karsilastirmasi kazanani
belirliyor. `nonlinear.csv` kasitli olarak dogrusal olmayan uretildi
ve **beklendigi gibi agac modeli kazandi** (RandomForest cv=0.753 vs
LogisticRegression cv=0.658).

**6. PCA kosullu ve dogru tetikleniyor.**
Sadece 30 ve 12 sayisal kolonlu veri setlerinde acildi. `sample.csv`'de
PCA=2 gorunuyor ama bu **kural degil strateji** kaynakli — iterasyon
2'de `pca_ters` acti (Bolum 10.7).

**7. Iterasyon sayisi anlamli degisiyor.**
Esigi gecen 5 veri seti 1 iterasyonda durdu (bos yere donmedi);
esigi gecemeyen `diabetes` ve `sample_reg` 3'e kadar gitti; `sample`
2. iterasyonda esigi gecip durdu. Dongu **kor kor donmuyor**.

---

## 14. Odev Kavramlari Karsilanma Tablosu

| # | Kavram | Durum | Nerede |
|---|---|---|---|
| 1 | Birkac model gelistirmek | ✅ **Karsilandi** | `modeler._candidate_models` — task basina 4 model (baseline + dogrusal + bagging + boosting), CV ile karsilastirma |
| 2 | Sonrasinda LLM'e gecmek | ✅ **Karsilandi** | `llm.py` + `planner`/`modeler` LLM katmani; `claude-sonnet-4-5` |
| 3 | **RAG optimizasyonu** | ❌ **Yok** | Asagida acikliyorum |
| 4 | **LLM model optimizasyonu** | ❌ **Yok** | Asagida acikliyorum |
| 5 | Log tutmasi | ✅ **Karsilandi** | `memory/logger.py` → `runs/<ts>/run.json` + `run.txt`; 48 run birikmis |
| 6 | Kendini iyilestirmesi | 🟡 **Kismi** | `orchestrator` self-improvement dongusu var ve calisiyor (`sample.csv` 0.33→1.00), **ama stratejiler zayif** (Bolum 10.6) |
| 7 | Veri onizleme | ✅ **Karsilandi** | Profil ciktisi: kolon listesi, dtype, tespit edilen tip, essiz sayi, null orani |
| 8 | Istatistik | ✅ **Karsilandi** | Betimsel istatistik tablosu (ort/std/min/q25/medyan/q75/max/carpiklik), kategorik dagilimlar, sinif dengesi, korelasyon analizi |
| 9 | Data preprocessing | ✅ **Karsilandi** | `preprocessor`: imputation, scaling, one-hot encoding, opsiyonel PCA — hepsi sklearn `Pipeline` icinde |
| 10 | Multi-agent otomasyon | ✅ **Karsilandi** | 8 ajan, ortak `Agent` arayuzu, `RunState` uzerinden gevsek bagli iletisim, `orchestrator` koordinasyonu |
| 11 | PCA | ✅ **Karsilandi** | `planner` kosullu karar + `preprocessor` uygulama; cancer ve nonlinear'da aktif |
| 12 | **Gini indeksi** | ✅ **Karsilandi** | Ucunun de ele alinmasi: impurity (sklearn varsayilani), importance (`_gini_importance`), katsayi (`_gini_coefficient`) — Bolum 8 |
| 13 | Preprocessing'ler (cogul) | ✅ **Karsilandi** | Sayisal ve kategorik icin ayri dallar, 2 imputation stratejisi, scaling, encoding, PCA |
| 14 | **Domain bagimsizlik** | ✅ **Karsilandi** | Bolum 4; kanit REPORT.md'de 7 veri seti / 6 farkli alan |

### 14.1 Yapilmayan Iki Madde — Durust Aciklama

**RAG optimizasyonu (yok).**
Sistemde vektor veritabani, embedding, dokuman parcalama (chunking)
veya benzerlik-tabanli **dokuman** getirme yok.

*En yakin sey:* `memory/store.py` gecmis run'lari getiriyor ve LLM
prompt'una baglam olarak ekliyor — bu bir "retrieval-augmented"
akistir, ama **klasik RAG degil**: metin embedding'i degil, yapisal
profil karsilastirmasi (3 sinyalli benzerlik skoru) kullaniyor.

*Neden yapilmadi:* Kavramin bu projede **ne uzerine** kurulacagi
belirsiz. Uc olasi yorum var ve ucu farkli isler:
1. ML dokumantasyonu uzerinde RAG (sklearn dokumanlarindan plan onerisi)
2. Gecmis run'lar uzerinde embedding-tabanli getirme (mevcut yapisal
   benzerligin yerine)
3. Veri setinin kendi icerigi uzerinde RAG (metin kolonlari icin)

**Hocaya sorulacak:** Hangisi kastediliyor? Yorum netlesmeden yapilan
is yanlis yone gidebilirdi.

**LLM model optimizasyonu (yok).**
Prompt optimizasyonu, few-shot ornek secimi, model karsilastirmasi
(sonnet vs opus), sicaklik/parametre ayari, token maliyeti optimizasyonu
veya fine-tuning yapilmadi. Tek model (`claude-sonnet-4-5`), tek
prompt, varsayilan parametreler.

*Neden yapilmadi:* Iki sebep. (a) Kavram belirsiz — "LLM'in kendisini
mi optimize edecegiz, yoksa LLM'i kullanma bicimini mi?" (b) Daha
onemlisi: optimizasyon **olcum** gerektirir, olcum icin de calisan bir
API anahtari gerekir. Ortamda anahtar olmadigi icin LLM'in etkisi hic
olculemedi — neyi optimize edecegimi bilemezdim.

**Savunmada tutumum:** Bu iki maddeyi eksik olarak kabul ediyorum,
gizlemeye calismiyorum. Ikisi de kapsam netlestikten sonra eklenebilir;
altyapi hazir (`llm.py` tek arayuz, `store.py` getirme mekanizmasi var).

---

## 15. Bilinen Sinirlar

Savunmada **sorulmasini beklemeden kendim soyleyecegim** liste:

1. **Metin kolonlari islenmiyor.** Tespit ediliyor ama atiliyor. NLP
   ozelligi (TF-IDF, embedding) yok.

2. **Tarih kolonlari islenmiyor** — ve **tespiti de dolayli.** `loader`
   `parse_dates` kullanmadigi icin CSV'deki tarih kolonlari string
   okunuyor; her satiri farkli oldugu icin `text` yoluna dusup
   atiliyorlar. Sonuc dogru (atiliyor) ama **mekanizma beklendigi gibi
   degil**: `datetime` dali gercek `datetime` dtype'li veride calisir,
   CSV'den gelen tarihte calismaz. `messy.csv`'deki `kayit_tarihi`
   `text` olarak isaretleniyor.

3. **Hiperparametre aramasi yok.** Modeller sabit varsayilanlarla
   deneniyor. Grid/random search veya Bayesian optimizasyon yok.
   Muhtemelen en buyuk skor kazanci burada.

4. **Clustering test edilmedi.** Hedef kolon yoksa `task_type`
   `"clustering"` oluyor ama bu yol icin **model havuzu ve
   degerlendirme mantigi yazilmadi.** `_candidate_models` clustering
   dalini icermiyor — pratikte bu yol calismaz.

5. **Aykiri deger (outlier) islemesi yok.** Carpiklik hesaplaniyor ve
   raporlaniyor ama hicbir karara baglanmiyor. Winsorization, IQR
   kirpma veya `RobustScaler` secenegi yok.

6. **Hedef kolon varsayimi kirilgan.** Belirtilmezse **son kolon**
   secilir. Bu yaygin bir konvansiyon ama garanti degil; hedefi ortada
   olan bir CSV'de yanlis kolon secilir. Kullanici `--target` ile
   duzeltebilir, ama sistem bunu kendisi fark edemez.

7. **Self-improvement stratejileri zayif ve sabit.** Sadece 3 strateji,
   ikisi belirli veri yapilarinda **yapisal olarak etkisiz**
   (Bolum 10.6). Ozellik secimi, sinif dengeleme (SMOTE), farkli
   olcekleme veya model hiperparametresi denenmiyor.

8. **PCA acikken feature importance yorumlanamaz.** Importance
   `pca0..pca9` bilesenleri uzerinden raporlaniyor, orijinal kolon
   adlari uzerinden degil. `cancer` ve `nonlinear` ciktilarinda bu
   gorunuyor. Kullanici "hangi olcum onemli?" sorusuna cevap alamiyor.

9. **LLM katmaninin faydasi olculmedi.** Gercek API anahtariyla
   calistirma yapilmadi. Kanitlanan tek sey: LLM yokken/bozukken sistem
   **zarar gormuyor**. LLM'in skoru **iyilestirdigi** gosterilmedi.

10. **Cok buyuk veri icin optimize degil.** Tum veri bellege okunuyor;
    permutation importance her calistirmada yeniden hesaplaniyor
    (n_repeats=5 × ozellik sayisi kadar yeniden tahmin).

11. **`REPORT.md` skorlari tek bir split'e dayaniyor.** `random_state=42`
    sabit, yani tekrarlanabilir — ama farkli bir seed farkli skor
    verebilir. Skorlarin guven araligi hesaplanmiyor.

12. **`orchestrator.py:13`'teki `PIPELINE` listesi olu kod.**
    Self-improvement eklendiginde `HAZIRLIK`/`DENEME`/`RECORDER`
    ayrimina gecildi; eski liste tanimli kaldi ama hicbir yerde
    kullanilmiyor. Zararsiz ama temizlenmeli.

---

## 16. Canli Demo Akisi

Toplam hedef: **~8-10 dakika**. Once terminali temizle, `data/` ve
`runs/` hazir olsun.

### Adim 0 — Hazirlik (demo oncesi, ekranda degil)

```bash
source .venv/bin/activate
python3 make_data.py     # 5 veri seti uretir, ~3 sn
```

### Adim 1 — Domain-bagimsizligin ilk gosterimi (~90 sn)

```bash
python3 -m automl.orchestrator --data data/iris.csv
```

**Dikkat cekilecek:**
- `Hedef: tur  Task: classification` — **hedef belirtmedim**, sistem
  son kolonu secti ve tipinden gorev tipini cikardi.
- Profil tablosu: her kolonun dtype'i ve **tespit edilen tipi**.
- `f1_weighted: 0.9333`

**Soylenecek cumle:** *"Bu komutta veri yolundan baska hicbir sey
vermedim. Simdi ayni komutu tamamen farkli bir veriyle calistiracagim."*

### Adim 2 — ⭐ EN ETKILI AN: Ayni komut, farkli task (~90 sn)

```bash
python3 -m automl.orchestrator --data data/diabetes.csv
```

**Dikkat cekilecek:**
- `Task: regression` — **komut ayni, tespit farkli.** Hedef kolonun
  tipi sayisal oldugu icin gorev tipi degisti.
- Metrikler degisti: `r2 / rmse / mae` (accuracy/f1 degil).
- Model havuzu degisti: `LinearRegression` var, `LogisticRegression` yok.

**Bu an sunumun en guclu noktasi.** Tek bir satir kod veya parametre
degismeden sistem tamamen farkli bir problem tipine gecti.

### Adim 3 — ⭐ PCA'nin otomatik acilmasi (~90 sn)

```bash
python3 -m automl.orchestrator --data data/cancer.csv
```

**Dikkat cekilecek:**
- `--- YUKSEK KORELASYONLAR ---` tablosu:
  `mean radius <-> mean perimeter r=+0.998`
- **Burada durakla ve acikla:** *"Cevre = 2πr. Bu iki kolon ayni bilgiyi
  tasiyor. 30 kolonun cogu bu sekilde birbirinin turevi."*
- `not: 30 sayisal kolon icin PCA acildi` → `X_train: (455, 30) -> (455, 10)`
- `gini: 0.9907` — Gini **katsayisi**
- `not: secilen model agac-bazli degil, Gini importance hesaplanamaz`

**Soylenecek:** *"PCA'yi kolon sayisi tetikledi, korelasyonlar degil —
ama korelasyonlar bu kararin neden dogru oldugunu gosteriyor."*

### Adim 4 — Gini karsilastirmasi (~60 sn)

```bash
python3 -m automl.orchestrator --data data/nonlinear.csv
```

**Dikkat cekilecek:**
- `RandomForest` kazandi (dogrusal olmayan veri)
- Hem `--- FEATURE IMPORTANCE ---` hem `--- GINI IMPORTANCE ---` dolu
- Onceki calistirmada (cancer) Gini importance **yoktu**

**Soylenecek:** *"Iste bu yuzden ana metrik permutation importance.
Gini importance sadece agac kazandiginda var; permutation her zaman var."*

### Adim 5 — ⭐ Zorlu veri ve drop kararlari (~90 sn)

```bash
python3 -m automl.orchestrator --data data/messy.csv
```

**Dikkat cekilecek:**
- Profil tablosunda: `bos_kolon ... bos: 100%`, `aciklama → text`,
  `urun_kodu → categorical (essiz: 60)`
- Plan notlari — **dort farkli kural, dort farkli sebep:**
  ```
  urun_kodu: 60 essiz kategori, atildi
  kayit_tarihi: text tipi, atildi
  aciklama: text tipi, atildi
  bos_kolon: %100 bos, atildi
  ```
- **Onemli:** `olcum_a` %15 eksik ama **atilmadi** — imputation ile
  kullanildi.

### Adim 6 — ⭐ Fallback'in devreye girmesi (~60 sn)

```bash
export ANTHROPIC_API_KEY="sk-ant-gecersiz"
python3 -m automl.orchestrator --data data/iris.csv
unset ANTHROPIC_API_KEY
```

**Dikkat cekilecek:**
- `! LLM cagrisi basarisiz: AuthenticationError: Error code: 401` — **iki kez**
  (planner + modeler)
- `kural-tabanli model havuzu kullanildi`
- `not: kural-tabanli plan kullanildi`
- **`f1_weighted: 0.9333` — skor degismedi**

**Soylenecek:** *"LLM yolu gercekten devreye girdi ve gercekten patladi.
Sistem cokmedi, skor degismedi. LLM bir ikinci gorus, tek karar verici degil."*

### Adim 7 — Testler (~30 sn)

```bash
python3 -m pytest tests/ -v
```

**Dikkat cekilecek:** `30 passed in ~2s`. Ozellikle
`test_scaler_test_setinden_etkilenmiyor` satirini isaret et ve
Bolum 6.1'deki senaryoyu bir cumleyle anlat.

### Adim 8 — Kapanis: karsilastirmali rapor (~60 sn)

```bash
python3 report.py
cat REPORT.md
```

**Dikkat cekilecek:** 7 veri seti, 6 farkli alan, 2 farkli task tipi,
6 satirdan 800 satira. **Tek kod tabani.**

**Kapanis cumlesi:** *"Bu tablo projenin iddiasinin kaniti. Simdi
sinirlarini da kendim soyleyeyim..."* → Bolum 15'ten 3-4 madde say.

### Demo Zaman Ozeti

| Adim | Sure | Etki |
|---|---|---|
| 1. iris | 90 sn | Isinma |
| 2. diabetes | 90 sn | ⭐ Task tespiti degisiyor |
| 3. cancer | 90 sn | ⭐ PCA otomatik aciliyor |
| 4. nonlinear | 60 sn | Gini karsilastirmasi |
| 5. messy | 90 sn | ⭐ Drop kararlari |
| 6. fallback | 60 sn | ⭐ Dayaniklilik |
| 7. testler | 30 sn | Guvence |
| 8. rapor | 60 sn | Kapanis |
| **Toplam** | **~9 dk** | |

---

## 17. Muhtemel Sorular ve Cevaplar

**S1. Esigi neden %5 sectin, keyfi degil mi?**
Kismen keyfi, ama davranisi sinirlarla guvence altina alinmis. %5 su
sezgiden geliyor: bir kolon satirlarin %5'inden az essiz deger
tasiyorsa muhtemelen kategori kodudur. Asil onemli olan **oran
kullanilmasi**, kesin degeri degil — `max(2, ...)` ve `min(20, ...)`
sinirlari uc durumlari zaten kesiyor. %3 veya %8 secseydim test
veri setlerinde sonuc degismezdi; sabit bir sayi secseydim degisirdi.
Ayarlanabilir olmasi bir iyilestirme olurdu.

**S2. Neden permutation importance, neden Gini?**
Iki sebep. Birincisi model-agnostiklik: Gini importance sadece agac
modellerinde var, sistemim hangi modelin kazanacagini bilmiyor.
REPORT.md'de 7 veri setinin 4'unde dogrusal model kazandi — ana metrik
Gini olsaydi bu 4 vakada importance tablosu bos kalirdi. Ikincisi
MDI'nin yuksek kardinaliteli ozellikleri sisirme yanliligi;
permutation test setinde gercek performans etkisini olctugu icin bu
yanliliktan muaf. **Yine de ikisini de hesapliyorum**, agac kazanirsa
her ikisi de raporlaniyor.

**S3. Sizinti olmadigindan nasil eminsin?**
Emin degilim, **test ediyorum**. `test_no_leakage.py`'de test setine
egitim verisinin ~20.000 kati buyuklukte uc degerler koyuyorum ve
scaler'in ogrendigi ortalamanin **49.5'te kaldigini** dogruluyorum.
Sizinti olsaydi ortalama milyonlara kayardi ve test kirmizi olurdu.
Ayri bir test imputer icin ayni seyi yapiyor, bir digeri de split'in
preprocess'ten once geldigini yapisal olarak dogruluyor.
Ayrica gercek veride scaler ortalamasinin `X_train`'e esit ve tum
veriye esit **olmadigini** kontrol ediyorum — ikisi tesadufen esit
olsaydi test hicbir sey kanitlamazdi.

**S4. Bu gercekten domain-bagimsiz mi, yoksa bu veri setlerine mi uydurdun?**
Uc somut argumanim var. (a) Kod tabaninda **tek bir kolon adi
gecmiyor** — `grep` ile gosterebilirim; planner sadece profildeki
ozelliklere bakiyor. (b) Veri setleri 6 farkli alandan: tibbi teshis,
botanik, tibbi ilerleme, perakende, emlak, sentetik; kolon adlari
Turkce ve Ingilizce karisik. (c) `messy.csv`'yi **sistem yazildiktan
sonra** ekledim ve tek satir kod degistirmeden dogru drop kararlarini
verdi. Ilk calistirmada bir cokme buldum ama o **raporlama katmaninda**
bir bicimlendirme hatasiydi (tamamen bos kolonda `None` istatistik),
karar mantiginda degil.
Durust sinir: veri setlerinin hicbirinde >50 kolonlu, milyonlarca
satirli veya cok dilli metin agirlikli veri yok.

**S5. Neden basit modeller kazandi, deep learning kullansaydin?**
Cunku bu veri setlerinde basit modeller **gercekten daha iyi**.
`cancer`'da LogisticRegression cv=0.980, RandomForest cv=0.947 —
dogrusal ayrilabilir bir problemde derin ag ekstra kazanc getirmez,
sadece asiri ogrenir. Derin ogrenme 569 satirda calismaz; binlerce
ornek ve GPU ister. Ayrica baseline karsilastirmasi bunu gosteriyor:
`cancer`'da baseline 0.482 → 0.980, yani problemin cogu zaten
dogrusal olarak cozuluyor. `nonlinear.csv`'yi tam da bunu test etmek
icin urettim: dogrusal olmayan yapida **agac modelleri kazandi**
(RF 0.753 vs LR 0.658). Yani sistem model secmiyor, **veri seciyor**.

**S6. Multi-agent demek icin sinif yazmak yeterli mi?**
Hayir, tek basina yeterli degil — ama burada sinifin otesinde uc
yapisal ozellik var. (a) **Gevsek baglilik:** ajanlar birbirini import
etmiyor, sadece `RunState` uzerinden konusuyor; `modeler`,
`planner`'in varligindan haberdar degil. (b) **Degistirilebilir
siralama:** orchestrator ajanlari `HAZIRLIK`/`DENEME` diye ikiye
bolup `DENEME`'yi tekrar tekrar calistirabiliyor — ajanlar buna adapte
oluyor cunku durum disarida. (c) **Bagimsiz test edilebilirlik:**
`test_no_leakage.py` sahte bir `RunState` kurup sadece `preprocessor`'u
calistiriyor.
Kabul ediyorum: bu **orkestre edilmis bir boru hatti**, ajanlarin
birbiriyle muzakere ettigi veya paralel calistigi bir sistem degil.
"Multi-agent" terimini bu anlamda kullaniyorum.

**S7. LLM olmadan da calisiyorsa LLM ne ise yariyor?**
Su anki durumda **kanitlanmis faydasi yok** — bunu acikca kabul
ediyorum. Gercek anahtarla test edemedigim icin LLM'in skoru
iyilestirdigini gosteremiyorum. Kanitladigim tek sey: LLM
yokken/bozukken sistem **zarar gormuyor**.
Tasarim gerekcesi su: kural-tabanli mantik sabit esiklerle calisir
(null > %50, essiz > 50, kolon > 10). LLM baglami gorup nuansli karar
verebilir — orn. "bu 60 essiz degerli kolon aslinda posta kodu, target
encoding daha uygun". Ama bu **iddia**, henuz **kanit** degil.
Bu yuzden LLM'i **ikinci gorus** olarak konumlandirdim, tek karar
verici degil.

**S8. Self-improvement gercekten iyilestiriyor mu, hangi ornekte skor artti?**
Evet, bir ornekte: **`sample.csv`**. Iterasyon 1 `varsayilan` ile
f1=0.3333, esigin (0.80) altinda kaldi; iterasyon 2 `pca_ters` ile
PCA acildi ve f1=1.0000'e cikti, esik gecildi, durdu.
**Durustluk kaydi:** `sample.csv` 6 satir, test seti ~2 satir. f1=1.0
"2/2 dogru" demek, istatistiksel olarak anlamsiz. Bu ornek dongunun
*mekanizmasinin* calistigini kanitlar, skorun gercekten iyi oldugunu
degil.
Ikinci kanit ters yonden: `sample_reg.csv`'de iterasyon 2 skoru
**-2.58**'e cakti ve sistem son iterasyonu degil **en iyisini**
(iterasyon 1, r2=0.1837) rapor etti.
Ve zayifligi da soyleyeyim: `diabetes`'te uc strateji de ayni skoru
verdi — sebebi matematiksel (Bolum 10.6).

**S9. Neden bu kadar cok `# type: ignore` var?**
Aslinda **sadece 4 tane** var, hepsini tek tek savunabilirim:
- `profiler.py:51` ve `:57` — `pandas.Series.std()` / `.skew()`
  donus tipi pyright icin `Series | float`; calisma zamaninda skalerdir.
  Pandas'in tip taniminin eksikligi, kodda sorun yok.
- `profiler.py:79` — `df[cols].corr()`; ortamda `pandas-stubs` kurulu
  olmadigi icin pyright `df[cols]`'u `DataFrame` degil `Series`
  sanip `corr()`'a eksik argüman diyor.
- `evaluator.py:96` — `permutation_importance` donusunun
  `importances_mean` alani stub'larda tanimsiz.
Dordu de **ucuncu parti tip taniminin eksikligi**, kendi kodumdaki bir
tip hatasi degil. `pyright automl` **0 hata** veriyor.

**S10. `diabetes`'te r2=0.4526 dusuk degil mi?**
Dusuk, ama bu **veri setinin dogasi**. Diabetes veri seti literaturde
bilinen zor bir regresyon problemi; dogrusal modellerle r2 genelde
0.4-0.5 bandinda. Baseline r2=-0.030, yani model gercekten bir sey
ogreniyor. Sistem bunu **fark etti** (esik 0.50'nin altinda) ve 3
iterasyon denedi; hicbiri iyilestiremedi ve durustce en iyisini
raporladi. Sistemin skoru sismedi — iyi bir isaret.

**S11. Neden `random_state=42` her yerde sabit?**
Tekrarlanabilirlik icin. Ayni komut ayni sonucu vermeli ki iterasyonlar
ve veri setleri karsilastirilabilsin. Ama bu bir **sinir**: skorlar tek
bir split'e dayaniyor, guven araligi yok. Farkli seed'lerle tekrarlayip
ortalama ve standart sapma raporlamak iyilestirme olurdu.

**S12. `messy.csv`'yi sen urettigin icin sonuc garanti degil mi?**
Kismen hakli bir elestiri. Ama iki savunmam var. (a) Veri setini
**sistem tamamlandiktan sonra** urettim ve ilk calistirmada gercek bir
cokme buldum — demek ki sonucu garantilememis. (b) Icindeki zorluklar
sentetik degil, **gercek dunyada yaygin** olanlar: eksik degerler,
yuksek kardinaliteli ID benzeri kolonlar, serbest metin, tamamen bos
kolonlar. Gercek bir kirli veri setiyle (orn. Titanic) test etmek daha
guclu olurdu; bunu bir sonraki adim olarak kabul ediyorum.

**S13. Bir kolon yanlis tiplenirse ne olur?**
Zincirleme etki olur, cunku task tespiti kolon tipine dayanir. Orn.
0/1 hedef kolonu yanlislikla `numeric` sayilirsa gorev **regresyona**
doner ve tum metrikler anlamsizlasir. Bu yuzden esikte `max(2, ...)`
alt siniri var — ikili hedefin kategorik kalmasini garantiler.
Kullanici `--target` ile hedefi duzeltebilir ama **tipi** elle
zorlayamaz; bu bir eksiklik.

**S14. Neden `f1_weighted`, neden `accuracy` degil?**
Accuracy dengesiz veride yanilticidir: %95'i tek siniftan olusan bir
veride hicbir sey ogrenmeyen model de %95 alir. `f1_weighted` her
sinifin precision/recall dengesini sinif buyuklugune gore
agirliklandirir. Ikisini de raporluyorum; `cancer`'da ikisi de 0.9737
cikti cunku o veri makul dengeli (%62.7 / %37.3).

**S15. PCA acikken ozellik onemi anlamsiz degil mi?**
Hakli, bu gercek bir sinir. `cancer` ve `nonlinear` ciktilarinda
importance `pca0..pca9` uzerinden raporlaniyor; "hangi olcum onemli?"
sorusuna cevap alinamiyor. Cozum PCA'nin `components_` matrisiyle geri
projeksiyon yapip orijinal kolonlara dagitmak olurdu — yapilmadi.
Bilinen sinirlar listesinde (Bolum 15, madde 8) yer aliyor.

**S16. Sistemin cokmedigi ne kadar test edildi?**
Uc kanit. (a) 30 test gecti, 11'i uctan uca `messy.csv` uzerinde.
(b) `report.py` 7 farkli veri setini pes pese isliyor, hicbiri
patlamiyor — 6 satirdan 800 satira. (c) Hata yollari acikca test
edildi: gecersiz LLM anahtari, bozuk JSON, olmayan hedef kolon,
tamamen bos kolon, cok kucuk veride CV.
Test edilmeyen: bozuk CSV, kodlama (encoding) hatalari, bellek
sinirini asan veri.

**S17. Kod kalitesi acisindan zayif noktalar neler?**
Kendim uc tane sayabilirim. (a) `orchestrator.py:13`'teki `PIPELINE`
listesi **olu kod** — self-improvement eklendiginde
`HAZIRLIK`/`DENEME` ayrimina gecildi, eski liste kaldi.
(b) `_yeterli_mi` fonksiyonu `metric_name` parametresi aliyor ama
kullanmiyor. (c) `orchestrator.run` fonksiyonu uzun (~100 satir) —
dongu ve raporlama kismi ayri fonksiyonlara bolunebilirdi.
Hicbiri islevi etkilemiyor ama temizlenmeleri gerekir.

**S18. Bu sistemi uretime alsan neyi once degistirirdin?**
Sirasiyla: (1) hiperparametre aramasi — muhtemelen en buyuk skor
kazanci; (2) tarih kolonlarini gercekten ayristirip yil/ay/gun/haftanin
gunu ozellikleri cikarmak; (3) capraz dogrulamayi tekrarli hale getirip
skorlara guven araligi eklemek; (4) PCA'li durumda importance'i
orijinal kolonlara geri projekte etmek. LLM optimizasyonu bu
listede **once gelmezdi** — once olculebilir kazanc, sonra LLM.

---

## Ek: Hizli Referans Kartlari

### Komutlar

```bash
python3 make_data.py                                    # veri uret
python3 -m automl.orchestrator --data data/iris.csv     # temel
python3 -m automl.orchestrator --data X --target hedef  # hedef belirt
python3 -m automl.orchestrator --data X --max-iterasyon 5
python3 -m pytest tests/ -v                             # 30 test
python3 report.py                                       # REPORT.md
python3 show_runs.py                                    # gecmis run'lar
.venv/bin/pyright automl                                # 0 hata
```

### Ezberlenecek Sayilar

| | |
|---|---|
| Ajan sayisi | 8 |
| Test sayisi | 30 (1.76 sn) |
| Veri seti | 7 (REPORT.md'de) |
| Kod satiri | ~2000 (testler dahil) |
| Tip esigi | `max(2, min(20, n_rows * 0.05))` |
| PCA kosulu | `numeric_cols > 10 AND n_rows > 50` |
| Esikler | classification 0.80, regression 0.50 |
| Korelasyon esigi | \|r\| >= 0.85 |
| Drop: null | > %50 |
| Drop: kardinalite | > 50 essiz |
| pyright | 0 hata |
| `# type: ignore` | 4 (hepsi ucuncu parti stub eksikligi) |

### Skorlar

| Veri | Task | Skor | Iterasyon |
|---|---|---|---|
| cancer | classification | f1 = 0.9737 | 1 |
| iris | classification | f1 = 0.9333 | 1 |
| messy | classification | f1 = 0.8625 | 1 |
| nonlinear | classification | f1 = 0.8433 | 1 |
| sample | classification | f1 = 1.0000 | 2 |
| diabetes | regression | r2 = 0.4526 | 3 |
| sample_reg | regression | r2 = 0.1837 | 3 |
