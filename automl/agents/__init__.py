"""Pipeline adimlari: her agent state alir, state dondurur."""
from automl.agents.base import Agent
from automl.agents.evaluator import EvaluatorAgent, evaluate
from automl.agents.loader import LoaderAgent, load
from automl.agents.modeler import ModelerAgent, train
from automl.agents.planner import PlannerAgent, plan
from automl.agents.preprocessor import PreprocessorAgent, preprocess
from automl.agents.profiler import ProfilerAgent, profile
from automl.agents.recorder import RecorderAgent, log
from automl.agents.splitter import SplitterAgent, split

__all__ = [
    "Agent",
    "LoaderAgent", "load",
    "ProfilerAgent", "profile",
    "SplitterAgent", "split",
    "PlannerAgent", "plan",
    "PreprocessorAgent", "preprocess",
    "ModelerAgent", "train",
    "EvaluatorAgent", "evaluate",
    "RecorderAgent", "log",
]
