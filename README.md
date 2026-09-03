# autoPilot — Domain-Bagimsiz AutoML Boru Hatti

Herhangi bir CSV dosyasini alip kolon tiplerini, hedef kolonu ve gorev
tipini (siniflandirma / regresyon) kendi tespit eden, preprocessing
planini otomatik uretip birden fazla modeli karsilastiran bir AutoML
sistemi. Veri setine ozel hicbir ayar gerekmez: ayni kod iris, kanser
teshisi, diyabet ilerlemesi ve eksik degerli karisik veride calisir.
Sonuc yeterince iyi degilse sistem farkli stratejiler deneyip kendini
iyilestirir.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Test veri setlerini uretmek icin:

```bash
python3 make_data.py
```

## Kullanim

```bash
# Temel kullanim: hedef kolon otomatik secilir (son kolon)
python3 -m automl.orchestrator --data data/iris.csv

# Hedef kolonu acikca belirtmek
python3 -m automl.orchestrator --data data/messy.csv --target hedef

# Self-improvement dongusunun iterasyon sayisini degistirmek (varsayilan 3)
python3 -m automl.orchestrator --data data/diabetes.csv --max-iterasyon 5
```

Yardimci komutlar:

```bash
python3 report.py           # tum veri setlerini calistirip REPORT.md uretir
python3 show_runs.py        # gecmis run'lari listeler
python3 -m pytest tests/ -v # testler
```

## Mimari

Sistem 8 ajandan olusur. Her ajan `RunState` adli tek bir "canta"
nesnesini alir, kendi alanini doldurur ve geri dondurur. Ajanlar
birbirini tanimaz; sadece `RunState` uzerinden konusurlar.

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

| Ajan | Sorumlulugu |
|---|---|
| `loader` | CSV'yi diskten okur |
| `profiler` | Kolon tipleri, hedef kolon, gorev tipi, istatistikler, korelasyonlar |
| `splitter` | Train/test ayrimi (preprocessing'den ONCE) |
| `planner` | Profile bakarak preprocessing plani uretir |
| `preprocessor` | Plani sklearn Pipeline'a cevirir, sadece train'de fit eder |
| `modeler` | Aday modelleri CV ile karsilastirir, en iyisini egitir |
| `evaluator` | Test skorlari, Gini katsayisi, feature importance |
| `recorder` | Run'i `runs/<timestamp>/` altina JSON + metin ozeti olarak yazar |

## Domain-bagimsizlik nasil saglaniyor

Sistemin hicbir yerinde kolon adi, veri seti adi veya alan bilgisi
sabit kodlanmis degildir. Uc mekanizma bunu saglar:

**1. Tip tespiti** (`profiler._detect_type`) — dtype'a korukorune
guvenmez, veriye bakar:
- Ondalik deger varsa kesin sayisal.
- Tamsayi kolonlarda esik **veri boyutuna gore** hesaplanir:
  `max(2, min(20, n_rows * 0.05))`. Boylece 8 essiz degerli bir kolon
  1000 satirlik veride kategorik, 40 satirlik veride sayisal sayilir.
  Sabit esik kullanilsaydi kucuk veride her sey kategorik gorunurdu.
- Her satirda farkli olan metin kolonlari `text` olarak isaretlenir ve
  ozellik olarak kullanilmaz.

**2. Task tespiti** — hedef kolonun tespit edilen tipi gorev tipini
belirler: sayisal ise regresyon, degilse siniflandirma. Hedef
belirtilmezse son kolon secilir. Olmayan bir hedef verilirse mevcut
kolonlari listeleyen acik bir hata firlatilir.

**3. Profile-gudumlu plan** — planner kolon adlarina degil, profile
cikan **ozelliklere** bakar: null orani %50'yi gecen kolonlar atilir,
50'den fazla essiz degerli kategorikler one-hot patlamasini onlemek
icin atilir, metin ve tarih kolonlari atilir, 10'dan fazla sayisal
kolon varsa PCA acilir.

`REPORT.md` bu iddianin kanitidir: ayni kod, farkli alanlardan yedi
veri setinde ayar degistirmeden calisir.

## Veri sizintisi onlemi

Bu sistemin en kritik garantisi: **preprocessor yalnizca train setinde
fit edilir.**

- `splitter` boru hattinda `preprocessor`dan once gelir, yani olcekleme
  ve imputation hesaplanirken test seti henuz gorulmemistir.
- `preprocessor` `fit_transform`'u sadece `X_train`'e, `transform`'u
  `X_test`'e uygular ([preprocessor.py](automl/agents/preprocessor.py)).
- Scaler'in ortalamasi/std'si ve imputer'in doldurma degeri train'den
  ogrenilir; test setindeki uc degerler bunlari etkilemez.

`tests/test_no_leakage.py` bunu dogrudan test eder: test setine
egitim verisinin ~20.000 kati buyuklukte uc degerler konur ve scaler'in
ogrendigi ortalamanin degismedigi dogrulanir. Sizinti olsaydi bu test
patlardi.

Ayrica iterasyonlar arasinda **split yeniden yapilmaz** — aksi halde
her denemede farkli bir test seti olusur ve iterasyonlarin skorlari
karsilastirilamaz hale gelirdi.

## LLM katmani ve fallback mantigi

`planner` ve `modeler` ajanlarinin uzerine opsiyonel bir LLM karar
katmani eklenmistir (Anthropic API, `claude-sonnet-4-5`).

Calisma sekli:
1. Kural-tabanli mantik **her zaman** once calisir ve guvenli tabani
   olusturur.
2. `ANTHROPIC_API_KEY` tanimliysa LLM'e de sorulur: profile ozeti,
   kural-tabanli planin onerisi ve benzer gecmis run'lar baglam olarak
   verilir.
3. LLM'in cevabi JSON olarak parse edilip pydantic semasiyla valide
   edilir.
4. **Uydurma kontrol:** LLM'in onerdigi kolon adlari gercekten veride
   var mi, onerdigi model adlari havuzda mi kontrol edilir. Uydurma
   varsa oneri tumden reddedilir.
5. Herhangi bir sorunda (anahtar yok, paket yok, ag hatasi, bozuk JSON,
   dogrulama hatasi, uydurma ad) sistem sessizce kural-tabanli yola
   duser ve calismaya devam eder.

`automl/llm.py` hicbir kosulda exception firlatmaz; her zaman ya gecerli
bir nesne ya da `None` dondurur. **LLM olmadan sistem tam islevseldir**
— testler de LLM kapaliyken calisir.

## Self-improvement dongusu

Sonuc yeterince iyi degilse sistem farkli bir strateji ile tekrar dener:

- **Esikler:** siniflandirmada `f1_weighted < 0.80`, regresyonda
  `r2 < 0.50` yetersiz sayilir.
- **Stratejiler:** 1) `varsayilan` (mevcut kurallar), 2) `pca_ters`
  (PCA aciksa kapat, kapaliysa ac), 3) `imputation_degis`
  (median→mean, most_frequent→constant).
- Tekrar denerken **sadece planner'dan itibaren** calisilir; veri
  okuma, profilleme ve split tekrarlanmaz.
- Her iterasyonun plani ve skoru kaydedilir; **son sonuc degil, en iyi
  skoru veren iterasyon** raporlanir.
- `--max-iterasyon` ile sinirlandirilir, sonsuz dongu olmaz.

Ayrica `planner` gecmis run'lardan ogrenir: `memory/store.py` benzer
profilli gecmis run'lari bulur, bunlar LLM'e baglam olarak verilir
(LLM yoksa bilgi amacli plan notlarina yazilir).

## Bilinen sinirlar

- **Metin ve tarih kolonlari islenmiyor.** Tespit ediliyorlar ama
  ozellik olarak kullanilmadan atiliyorlar. NLP ozellikleri veya
  tarihten yil/ay/gun cikarimi yok.
- **Tarih kolonlari CSV'den string olarak okunuyor.** `loader` tarih
  ayristirmasi yapmadigi icin bir tarih kolonu genellikle `datetime`
  degil `text` olarak isaretlenir. Sonuc ayni (atiliyor) ama sebep
  farkli.
- **Hiperparametre aramasi yok.** Modeller sabit varsayilan
  parametrelerle denenir; grid/random search uygulanmaz.
- **Clustering test edilmedi.** Hedef kolon yoksa `task_type`
  `clustering` olur ama bu yol icin model havuzu ve degerlendirme
  mantigi yazilmadi.
- **Self-improvement stratejileri sabit ve dar.** Uc strateji vardir;
  ozellik secimi, farkli olcekleme veya sinif dengeleme denenmez.
- **Cok buyuk veri icin optimize degil.** Tum veri bellege okunur,
  permutation importance her calistirmada yeniden hesaplanir.
