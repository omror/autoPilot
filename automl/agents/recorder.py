"""Run'i diske kaydeder."""
from automl.agents.base import Agent
from automl.schemas import RunState


def log(state: RunState) -> RunState:
    """Run'i diske kaydeder."""
    from automl.memory.logger import save_run

    sure = getattr(state, "sure_sn", 0.0)
    klasor = save_run(state, sure)
    print(f"   kaydedildi: {klasor}")
    return state


class RecorderAgent(Agent):
    """Run'i diske kaydeder."""

    name = "recorder"

    def run(self, state: RunState) -> RunState:
        return log(state)
