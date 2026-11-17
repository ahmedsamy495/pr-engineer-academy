from src.evaluators.deterministic import (
    JsonSchemaEvaluator,
    SentimentAccuracyEvaluator,
    LatencyBudgetEvaluator,
)
from src.evaluators.llm_judge import LLMJudgeEvaluator
from src.models import ExecutionResult, TestCase


def test_json_schema_evaluator_valid():
    evaluator = JsonSchemaEvaluator(required_keys=["sentiment", "confidence_score"])
    execution = ExecutionResult(
        prompt_name="p",
        test_case_id="c1",
        rendered_prompt="test",
        raw_response='{"sentiment": "POSITIVE", "confidence_score": 0.95}',
        latency_ms=100.0,
        token_estimate=20,
    )
    case = TestCase(id="c1", input_vars={})
    result = evaluator.evaluate(execution, case)

    assert result.passed is True
    assert result.score == 1.0


def test_json_schema_evaluator_invalid_json():
    evaluator = JsonSchemaEvaluator(required_keys=["sentiment"])
    execution = ExecutionResult(
        prompt_name="p",
        test_case_id="c1",
        rendered_prompt="test",
        raw_response="This is plain text with no json.",
        latency_ms=100.0,
        token_estimate=20,
    )
    case = TestCase(id="c1", input_vars={})
    result = evaluator.evaluate(execution, case)

    assert result.passed is False
    assert result.score == 0.0


def test_sentiment_accuracy_evaluator():
    evaluator = SentimentAccuracyEvaluator()
    case = TestCase(
        id="c1",
        input_vars={},
        expected={"sentiment": "NEGATIVE"},
    )
    matching_execution = ExecutionResult(
        prompt_name="p",
        test_case_id="c1",
        rendered_prompt="test",
        raw_response='{"sentiment": "NEGATIVE"}',
        latency_ms=80.0,
        token_estimate=15,
    )
    res = evaluator.evaluate(matching_execution, case)
    assert res.passed is True
    assert res.score == 1.0


def test_latency_budget_evaluator():
    evaluator = LatencyBudgetEvaluator(max_latency_ms=300.0)
    fast_exec = ExecutionResult(
        prompt_name="p",
        test_case_id="c1",
        rendered_prompt="test",
        raw_response="ok",
        latency_ms=120.0,
        token_estimate=5,
    )
    case = TestCase(id="c1", input_vars={})
    res = evaluator.evaluate(fast_exec, case)
    assert res.passed is True
