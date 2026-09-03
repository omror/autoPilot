"""Tüm agent'ların ortak arayüzü"""
from abc import ABC, abstractmethod
from automl.schemas import RunState


class Agent(ABC):
    """Her agent bir aşamadan sorumludur: state alır, state dondurur."""

    name: str = "agent"

    @abstractmethod
    def run(self, state: RunState) -> RunState:
        ...

    def __call__(self, state: RunState) -> RunState:
        return self.run(state)