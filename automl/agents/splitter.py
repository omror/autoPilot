"""Train/test ayrimi."""
import pandas as pd
from sklearn.model_selection import train_test_split
from automl.agents.base import Agent
from automl.schemas import RunState


def _tabakalama_etiketi(y: pd.Series) -> pd.Series:
    """train_test_split'e verilecek stratify etiketi.

    Tabakalama her sinifta en az 2 ornek ister. Tek ornekli siniflar
    tabakalamayi TUM veri icin kapatmasin diye en kalabalik sinifa katilir:
    kendileri tabakalanamaz ama diger siniflarin orani korunur. Modeler'daki
    StratifiedKFold da tek ornekli sinifi ayni sekilde tolere eder.
    """
    sayim = y.value_counts()
    tekil = [sinif for sinif, n in sayim.items() if n < 2]
    if not tekil:
        return y
    return y.where(~y.isin(tekil), sayim.index[0])


def split(state: RunState) -> RunState:
    "Train/test ayrımı, Preprocessing'den önce yapılır."
    df = state.df
    p = state.profile
    if p is None:
        raise RuntimeError("split: once profile adimi calismali")

    if p.target is None:
        state.X_train = df
        state.X_test = df.iloc[0:0]
        return state

    X = df.drop(columns = [p.target])
    y = df[p.target]

    stratify = None
    if p.task_type == "classification":
        stratify = _tabakalama_etiketi(y)

    try: 
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size = 0.2, random_state = 42, stratify = stratify
        )

    except ValueError as e:
        # Orn. sinif sayisi test setinin satir sayisindan fazla.
        if stratify is not None:
            print(f"   ! tabakali bolme yapilamadi, tabakasiz bolundu: {e}")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size = 0.2, random_state=42
        )

    state.X_train, state.X_test = X_tr, X_te
    state.y_train, state.y_test = y_tr, y_te
    return state


class SplitterAgent(Agent):
    """Train/test ayrimini yapar."""

    name = "splitter"

    def run(self, state: RunState) -> RunState:
        return split(state)
