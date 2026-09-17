"""Aday modelleri CV ile karsilastirir ve en iyisini egitir."""
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from automl import llm
from automl.agents.base import Agent
from automl.agents.profiler import ana_metrik
from automl.schemas import ModelScore, ModelSecimi, RunState

SYSTEM_PROMPT = (
    "Sen bir ML model secimi uzmanisin. Verilen veri profiline bakarak "
    "hangi aday modellerin denenmesi gerektigini oner."
)


# Baseline her denemede CV'ye girer: secilen modelin ondan ne kadar iyi
# oldugu evaluator'da raporlanir, LLM havuzu daraltsa bile.
BASELINE = "Baseline"


def _candidate_models(task_type: str, n_rows:int,
                      dengesiz: bool = False) -> dict:
    """Task tipine göre aday model havuzunu otomatik seçer.

    Dengesiz classification verisinde class_weight='balanced' varyantlari
    havuza EKLENIR; mevcut modeller de kalir ki CV ikisini karsilastirsin.
    Dengeli veride havuz aynen korunur.
    """
    if task_type == "regression":
        return {
            "Baseline": DummyRegressor(strategy="mean"),
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=100, random_state=42
            ),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),

        }
    models = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        }
    if dengesiz:
        # Azinlik sinifindaki hatayi sinif frekansinin tersiyle agirliklandirir.
        models["LogisticRegression_balanced"] = LogisticRegression(
            max_iter=1000, class_weight="balanced"
        )
        models["RandomForest_balanced"] = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )
    return models

def _cv_bolucu(task_type: str, y) -> StratifiedKFold | int:
    """Cross-validation fold bolucusu.

    Classification: StratifiedKFold, her fold verideki sinif oranlarini
    korur; dengesiz veride azinlik sinifi her fold'a dagilir. shuffle +
    random_state ile bolme tekrarlanabilir.
    Regression: fold sayisi (int) doner, sklearn tabakasiz KFold kurar.
    """
    #Küçük veride fold sayısını otomatik düşürür
    n_splits = max(2, min(5, len(y) // 2))
    if task_type != "classification":
        return n_splits
    # Fold sayisi en az ornekli sinifi gecmesin: her fold o siniftan ornek
    # alabilsin. Tek ornekli sinifta 2 fold'a iner (sklearn uyari verir).
    n_splits = max(2, min(n_splits, int(y.value_counts().min())))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def train(state: RunState,
          izinli_modeller: list[str] | None = None) -> RunState:
    "Aday modelleri CV ile karşılaştırır, en iyisini seçip eğitir."
    p = state.profile
    if p is None:
        raise RuntimeError("profile üretilmemiş, önce profiler çalışmalı")
    X, y = state.X_train_t, state.y_train

    models = _candidate_models(p.task_type, p.n_rows, p.is_imbalanced)
    if izinli_modeller:
        models = {k: v for k, v in models.items()
                  if k in izinli_modeller or k == BASELINE}
    # Dengesiz veride f1_macro: azinlik sinifi skora esit agirlikla girer.
    scoring = ana_metrik(p)

    cv_bolucu = _cv_bolucu(p.task_type, y)

    scores = []
    for name, model in models.items():
        try:
            cv = cross_val_score(model, X, y, cv=cv_bolucu, scoring=scoring)
            scores.append(
                ModelScore(
                    name = name,
                    cv_mean = float(np.mean(cv)),
                    cv_std = float(np.std(cv)),
                    )
            )
            
            
        
        except Exception as e:
            print(f"    ! {name} atlandı: {type(e).__name__}: {e}")


    if not scores:
        raise RuntimeError("Hiçbir model eğitilmedi")

    best = max(scores, key=lambda s: s.cv_mean)
    best_model = models[best.name]
    best_model.fit(X, y)

    state.candidates = scores
    state.best_name = best.name
    state.model = best_model
    print(f"    secilen model: {best.name} (cv={best.cv_mean:.3f})")
    return state


def _model_ozeti(state: RunState, havuz: list[str]) -> str:
    "LLM'e verilecek profil + mevcut model havuzu ozeti."
    p = state.profile
    assert p is not None
    satirlar = [
        f"Satir sayisi: {p.n_rows}",
        f"Kolon sayisi: {p.n_cols}",
        f"Gorev tipi: {p.task_type}",
        f"Hedef kolon: {p.target}",
    ]
    if p.class_balance:
        satirlar.append(f"Sinif dagilimi: {p.class_balance}")
    if p.is_imbalanced:
        satirlar.append(
            f"Dengesiz veri: azinlik sinifi {p.minority_class!r} "
            f"(oran {p.minority_ratio:.4f}), CV metrigi {ana_metrik(p)}"
        )
    if p.high_correlations:
        satirlar.append(
            f"Yuksek korelasyonlu cift sayisi: {len(p.high_correlations)}"
        )

    pl = state.plan
    if pl is not None:
        satirlar.append(
            f"Ozellik sayisi: {len(pl.numeric_cols)} sayisal, "
            f"{len(pl.categorical_cols)} kategorik"
        )
        satirlar.append(f"PCA: {pl.use_pca} ({pl.n_components})")

    satirlar += [
        "",
        f"Mevcut model havuzu: {havuz}",
        "",
        "Hangi modellerin denenmesi gerektigini JSON olarak dondur. "
        "Sadece JSON dondur, baska aciklama yazma. Alanlar: models "
        "(model adlarindan olusan liste), reason (kisa gerekce). "
        "SADECE yukaridaki havuzda gecen adlari kullan, yeni model adi "
        "uydurma.",
    ]
    return "\n".join(satirlar)


def _secim_gecerli(secim: ModelSecimi, havuz: list[str]) -> bool:
    "LLM'in sectigi model adlari gercekten havuzda var mi?"
    for ad in secim.models:
        if ad not in havuz:
            print(f"   ! LLM uydurma model adi verdi: {ad!r}, secim reddedildi")
            return False
    if not secim.models:
        print("   ! LLM bos model listesi verdi, secim reddedildi")
        return False
    return True


class ModelerAgent(Agent):
    """Aday modelleri karsilastirir, en iyisini egitir."""

    name = "modeler"

    def run(self, state: RunState) -> RunState:
        p = state.profile
        if p is None:
            raise RuntimeError("profile üretilmemiş, önce profiler çalışmalı")

        havuz = list(_candidate_models(p.task_type, p.n_rows,
                                       p.is_imbalanced).keys())

        # LLM varsa havuzu daraltmasini iste, yoksa tum havuz denenir.
        secim = None
        if llm.is_available():
            secim = llm.ask_json(
                SYSTEM_PROMPT,
                _model_ozeti(state, havuz),
                ModelSecimi,
            )

        if secim is not None and _secim_gecerli(secim, havuz):
            print(f"    LLM model secimi: {secim.models}")
            if secim.reason:
                print(f"    gerekce: {secim.reason}")
            return train(state, izinli_modeller=secim.models)

        print("    kural-tabanli model havuzu kullanildi")
        return train(state)
