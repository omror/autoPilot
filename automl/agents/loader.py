"""Veri setini diskten okur."""
import pandas as pd
from automl.agents.base import Agent
from automl.schemas import RunState


def load(state: RunState) -> RunState:
    state.df = pd.read_csv(state.data_path)
    return state


class LoaderAgent(Agent):
    """Veriyi diskten okur."""

    name = "loader"

    def run(self, state: RunState) -> RunState:
        return load(state)
