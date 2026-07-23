"""
src/ragas_evaluation/__init__.py
"""

__all__ = ["RagasEvaluator", "RagasResult"]

from src.ragas_evaluation.ragas_evaluator import RagasEvaluator, RagasResult


def __getattr__(name):
    if name in __all__:
        from src.ragas_evaluation.ragas_evaluator import RagasEvaluator, RagasResult
        return {"RagasEvaluator": RagasEvaluator, "RagasResult": RagasResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")