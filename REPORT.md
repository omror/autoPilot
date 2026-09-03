# Karsilastirmali Rapor

Asagidaki tablo tek bir kod tabaninin, veri setine ozel hicbir ayar yapilmadan farkli alanlardan gelen verilerde calistigini gosterir. Task tipi, kolon tipleri ve preprocessing plani her veri icin otomatik tespit edilmistir.

| veri seti | satir x kolon | task | atilan kolon | PCA | model | metrik | skor | iterasyon | sure |
|---|---|---|---|---|---|---|---|---|---|
| cancer.csv | 569 x 31 | classification | 0 | 10 | LogisticRegression | f1_weighted | 0.9737 | 1 | 0.7 sn |
| diabetes.csv | 442 x 11 | regression | 0 | - | LinearRegression | r2 | 0.4526 | 3 | 2.0 sn |
| iris.csv | 150 x 5 | classification | 0 | - | LogisticRegression | f1_weighted | 0.9333 | 1 | 0.6 sn |
| messy.csv | 400 x 9 | classification | 4 | - | GradientBoosting | f1_weighted | 0.8625 | 1 | 0.5 sn |
| nonlinear.csv | 800 x 13 | classification | 0 | 10 | RandomForest | f1_weighted | 0.8433 | 1 | 1.2 sn |
| sample.csv | 6 x 4 | classification | 0 | 2 | GradientBoosting | f1_weighted | 1.0000 | 2 | 0.3 sn |
| sample_reg.csv | 8 x 4 | regression | 0 | - | LinearRegression | r2 | 0.1837 | 3 | 0.5 sn |

Uretim zamani: 2026-09-03 13:46:35
