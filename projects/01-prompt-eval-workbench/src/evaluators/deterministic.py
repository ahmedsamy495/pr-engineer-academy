import json
import re
from typing import List, Optional
from src.evaluators.base import BaseEvaluator
from src.models import ExecutionResult, TestCase, EvalMetricResult


class JsonSchemaEvaluator(BaseEvaluator):
    """Checks whether the LLM output is valid JSON and matches expected keys/types."""

    name = "json_schema_validity"

    def __init__(self, required_keys: Optional[List[str]] = None):
        self.required_keys = required_keys or ["sentiment", "confidence_score"]

    def evaluate(self, execution: ExecutionResult, test_case: TestCase) -> EvalMetricResult:
        raw = execution.raw_response.strip()

        # Try to extract JSON if wrapped in markdown code fence ```json ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return EvalMetricResult(
                    metric_name=self.name,
                    score=0.0,
                    passed=False,
                    details={"error": "Output is valid JSON but not a JSON object."},
                )

            missing_keys = [k for k in self.required_keys if k not in data]
            if missing_keys:
                return EvalMetricResult(
                    metric_name=self.name,
                    score=0.5,
                    passed=False,
                    details={"missing_keys": missing_keys, "parsed_data": data},
                    reasoning=f"Missing required JSON keys: {missing_keys}",
                )

            return EvalMetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                details={"parsed_data": data},
                reasoning="Strict JSON parsed successfully with all required schema fields.",
            )

        except json.JSONDecodeError as e:
            return EvalMetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                details={"error": str(e), "raw_snippet": raw[:120]},
                reasoning="Response is not valid JSON.",
            )


class SentimentAccuracyEvaluator(BaseEvaluator):
    """Compares the extracted sentiment against ground-truth expected value."""

    name = "sentiment_accuracy"

    def evaluate(self, execution: ExecutionResult, test_case: TestCase) -> EvalMetricResult:
        if not test_case.expected or "sentiment" not in test_case.expected:
            return EvalMetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                details={"note": "No expected sentiment ground-truth provided."},
            )

        expected_sentiment = str(test_case.expected["sentiment"]).upper()
        raw = execution.raw_response.upper()

        # Look for JSON value first
        try:
            data = json.loads(execution.raw_response.strip())
            if isinstance(data, dict) and "sentiment" in data:
                actual = str(data["sentiment"]).upper()
                is_match = actual == expected_sentiment
                return EvalMetricResult(
                    metric_name=self.name,
                    score=1.0 if is_match else 0.0,
                    passed=is_match,
                    details={"expected": expected_sentiment, "actual": actual},
                    reasoning=f"Matched: {is_match} (Expected {expected_sentiment}, got {actual})",
                )
        except Exception:
            pass

        # Fallback to regex text search
        is_match = expected_sentiment in raw
        return EvalMetricResult(
            metric_name=self.name,
            score=1.0 if is_match else 0.0,
            passed=is_match,
            details={"expected": expected_sentiment, "found_in_text": is_match},
            reasoning=f"Found '{expected_sentiment}' in unstructured text: {is_match}",
        )


class LatencyBudgetEvaluator(BaseEvaluator):
    """Enforces a maximum latency budget in milliseconds."""

    name = "latency_budget"

    def __init__(self, max_latency_ms: float = 1500.0):
        self.max_latency_ms = max_latency_ms

    def evaluate(self, execution: ExecutionResult, test_case: TestCase) -> EvalMetricResult:
        passed = execution.latency_ms <= self.max_latency_ms
        score = min(1.0, self.max_latency_ms / max(1.0, execution.latency_ms))
        return EvalMetricResult(
            metric_name=self.name,
            score=round(score, 2),
            passed=passed,
            details={"latency_ms": execution.latency_ms, "budget_ms": self.max_latency_ms},
            reasoning=f"Latency {execution.latency_ms}ms vs budget {self.max_latency_ms}ms",
        )
