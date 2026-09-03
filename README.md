# autoPilot — Domain Bağımsız AutoML Boru Hattı

Herhangi bir CSV dosyasını alıp kolon tiplerini, hedef kolonu ve görev
tipini (sınıflandırma / regresyon) kendi tespit eden, preprocessing
planını otomatik üretip birden fazla modeli karşılaştıran bir AutoML
sistemi. Veri setine özel hiçbir ayar gerekmez: aynı kod iris, kanser
teşhisi, diyabet ilerlemesi ve eksik değerli karışık veride çalışır.
Sonuç yeterince iyi değilse sistem farklı stratejiler deneyip kendini
iyileştirir.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Test veri setlerini üretmek için:

```bash
python3 make_data.py
```

## Kullanım

```bash
# Temel kullanım: hedef kolon otomatik seçilir (son kolon)
python3 -m automl.orchestrator --data data/iris.csv

# Hedef kolonu açıkça belirtmek
python3 -m automl.orchestrator --data data/messy.csv --target hedef

# Self-improvement döngüsünün iterasyon sayısını değiştirmek (varsayılan 3)
python3 -m automl.orchestrator --data data/diabetes.csv --max-iterasyon 5
```

Yardımcı komutlar:

```bash
python3 report.py           # tüm veri setlerini çalıştırıp REPORT.md üretir
python3 show_runs.py        # geçmiş run'ları listeler
python3 -m pytest tests/ -v # testler
```

## Mimari

Sistem 8 ajandan oluşur. Her ajan `RunState` adlı tek bir "çanta"
nesnesini alır, kendi alanını doldurur ve geri döndürür. Ajanlar
birbirini tanımaz; sadece `RunState` üzerinden konuşurlar.

```
                    ┌─────────────── RunState ───────────────┐
                    │  df, X_train, plan, model, result ...  │
                    └────────────────────────────────────────┘
                              ▲ her ajan okur/yazar

  HAZIRLIK (bir kez çalışır)
  ┌─────────┐   ┌──────────┐   ┌──────────┐
  │ loader  │──▶│ profiler │──▶│ splitter │
  └─────────┘   └──────────┘   └──────────┘
   CSV oku      tip + task     train/test ayır
                tespiti        (sızıntı önlemi)
                                     │
        ┌────────────────────────────┘
        ▼
  DENEME (her iterasyonda tekrar çalışır)
  ┌─────────┐   ┌──────────────┐   ┌─────────┐   ┌───────────┐
  │ planner │──▶│ preprocessor │──▶│ modeler │──▶│ evaluator │
  └─────────┘   └──────────────┘   └─────────┘   └───────────┘
   plan üret     Pipeline kur       CV ile        test skoru +
   (+LLM)        fit SADECE train   model seç     importance
        ▲                                              │
        │      skor eşiğin altındaysa                  │
        └──────── strateji değiştir, tekrar ───────────┘
                                                       ▼
                                                 ┌──────────┐
                                                 │ recorder │
                                                 └──────────┘
                                                  runs/ altına kaydet
```

| Ajan | Sorumluluğu |
|---|---|
| `loader` | CSV'yi diskten okur |
| `profiler` | Kolon tipleri, hedef kolon, görev tipi, istatistikler, korelasyonlar |
| `splitter` | Train/test ayrımı (preprocessing'den ÖNCE) |
| `planner` | Profile bakarak preprocessing planı üretir |
| `preprocessor` | Planı sklearn Pipeline'a çevirir, sadece train'de fit eder |
| `modeler` | Aday modelleri CV ile karşılaştırır, en iyisini eğitir |
| `evaluator` | Test skorları, Gini katsayısı, feature importance |
| `recorder` | Run'ı `runs/<timestamp>/` altına JSON + metin özeti olarak yazar |

## Domain bağımsızlık nasıl sağlanıyor

Sistemin hiçbir yerinde kolon adı, veri seti adı veya alan bilgisi
sabit kodlanmış değildir. Üç mekanizma bunu sağlar:

**1. Tip tespiti** (`profiler._detect_type`) — dtype'a doğrudan
güvenmez, veriye bakar:
- Ondalık değer varsa kesin sayısal.
- Tamsayı kolonlarda eşik **veri boyutuna göre** hesaplanır:
  `max(2, min(20, n_rows * 0.05))`. Böylece 8 eşsiz değerli bir kolon
  1000 satırlık veride kategorik, 40 satırlık veride sayısal sayılır.
  Sabit eşik kullanılsaydı küçük veride her şey kategorik görünürdü.
- Her satırda farklı olan metin kolonları `text` olarak işaretlenir ve
  özellik olarak kullanılmaz.

**2. Task tespiti** — hedef kolonun tespit edilen tipi görev tipini
belirler: sayısal ise regresyon, değilse sınıflandırma. Hedef
belirtilmezse son kolon seçilir. Olmayan bir hedef verilirse mevcut
kolonları listeleyen açık bir hata fırlatılır.

**3. Profile güdümlü plan** — planner kolon adlarına değil, profilden
çıkan **özelliklere** bakar: null oranı %50'yi geçen kolonlar atılır,
50'den fazla eşsiz değerli kategorikler one-hot patlamasını önlemek
için atılır, metin ve tarih kolonları atılır, 10'dan fazla sayısal
kolon varsa PCA açılır.

`REPORT.md` bu iddianın kanıtıdır: aynı kod, farklı alanlardan yedi
veri setinde ayar değiştirmeden çalışır.

## Veri sızıntısı önlemi

Bu sistemin en kritik garantisi: **preprocessor yalnızca train setinde
fit edilir.**

- `splitter` boru hattında `preprocessor`dan önce gelir, yani ölçekleme
  ve imputation hesaplanırken test seti henüz görülmemiştir.
- `preprocessor` `fit_transform`'u sadece `X_train`'e, `transform`'u
  `X_test`'e uygular ([preprocessor.py](automl/agents/preprocessor.py)).
- Scaler'ın ortalaması/std'si ve imputer'ın doldurma değeri train'den
  öğrenilir; test setindeki uç değerler bunları etkilemez.

`tests/test_no_leakage.py` bunu doğrudan test eder: test setine
eğitim verisinin ~20.000 katı büyüklükte uç değerler konur ve scaler'ın
öğrendiği ortalamanın değişmediği doğrulanır. Sızıntı olsaydı bu test
patlardı.

Ayrıca iterasyonlar arasında **split yeniden yapılmaz** — aksi halde
her denemede farklı bir test seti oluşur ve iterasyonların skorları
karşılaştırılamaz hale gelirdi.

## LLM katmanı ve fallback mantığı

`planner` ve `modeler` ajanlarının üzerine opsiyonel bir LLM karar
katmanı eklenmiştir (Anthropic API, `claude-sonnet-4-5`).

Çalışma şekli:
1. Kural tabanlı mantık **her zaman** önce çalışır ve güvenli tabanı
   oluşturur.
2. `ANTHROPIC_API_KEY` tanımlıysa LLM'e de sorulur: profil özeti,
   kural tabanlı planın önerisi ve benzer geçmiş run'lar bağlam olarak
   verilir.
3. LLM'in cevabı JSON olarak parse edilip pydantic şemasıyla valide
   edilir.
4. **Uydurma kontrolü:** LLM'in önerdiği kolon adları gerçekten veride
   var mı, önerdiği model adları havuzda mı kontrol edilir. Uydurma
   varsa öneri tümden reddedilir.
5. Herhangi bir sorunda (anahtar yok, paket yok, ağ hatası, bozuk JSON,
   doğrulama hatası, uydurma ad) sistem sessizce kural tabanlı yola
   düşer ve çalışmaya devam eder.

`automl/llm.py` hiçbir koşulda exception fırlatmaz; her zaman ya geçerli
bir nesne ya da `None` döndürür. **LLM olmadan sistem tam işlevseldir**
— testler de LLM kapalıyken çalışır.

## Self-improvement döngüsü

Sonuç yeterince iyi değilse sistem farklı bir strateji ile tekrar dener:

- **Eşikler:** sınıflandırmada `f1_weighted < 0.80`, regresyonda
  `r2 < 0.50` yetersiz sayılır.
- **Stratejiler:** 1) `varsayilan` (mevcut kurallar), 2) `pca_ters`
  (PCA açıksa kapat, kapalıysa aç), 3) `imputation_degis`
  (median→mean, most_frequent→constant).
- Tekrar denerken **sadece planner'dan itibaren** çalışılır; veri
  okuma, profilleme ve split tekrarlanmaz.
- Her iterasyonun planı ve skoru kaydedilir; **son sonuç değil, en iyi
  skoru veren iterasyon** raporlanır.
- `--max-iterasyon` ile sınırlandırılır, sonsuz döngü olmaz.

Ayrıca `planner` geçmiş run'lardan öğrenir: `memory/store.py` benzer
profilli geçmiş run'ları bulur, bunlar LLM'e bağlam olarak verilir
(LLM yoksa bilgi amaçlı plan notlarına yazılır).

## Bilinen sınırlar

- **Metin ve tarih kolonları işlenmiyor.** Tespit ediliyorlar ama
  özellik olarak kullanılmadan atılıyorlar. NLP özellikleri veya
  tarihten yıl/ay/gün çıkarımı yok.
- **Tarih kolonları CSV'den string olarak okunuyor.** `loader` tarih
  ayrıştırması yapmadığı için bir tarih kolonu genellikle `datetime`
  değil `text` olarak işaretlenir. Sonuç aynı (atılıyor) ama sebep
  farklı.
- **Hiperparametre araması yok.** Modeller sabit varsayılan
  parametrelerle denenir; grid/random search uygulanmaz.
- **Clustering test edilmedi.** Hedef kolon yoksa `task_type`
  `clustering` olur ama bu yol için model havuzu ve değerlendirme
  mantığı yazılmadı.
- **Self-improvement stratejileri sabit ve dar.** Üç strateji vardır;
  özellik seçimi, farklı ölçekleme veya sınıf dengeleme denenmez.
- **Çok büyük veri için optimize değil.** Tüm veri belleğe okunur,
  permutation importance her çalıştırmada yeniden hesaplanır.
