from abc import ABC, abstractmethod
from src.models import ExecutionResult, TestCase, EvalMetricResult


class BaseEvaluator(ABC):
    """Abstract base class for all benchmark evaluators."""

    name: str = "base_evaluator"

    @abstractmethod
    def evaluate(self, execution: ExecutionResult, test_case: TestCase) -> EvalMetricResult:
        """Evaluate a single execution result against test case requirements."""
        pass
