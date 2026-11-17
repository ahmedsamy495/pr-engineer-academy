from src.runner import BenchmarkRunner
from src.providers.mock import MockProvider
from src.evaluators.deterministic import JsonSchemaEvaluator, SentimentAccuracyEvaluator
from src.models import PromptTemplate, TestCase


def test_runner_flow():
    provider = MockProvider()
    evaluators = [
        JsonSchemaEvaluator(required_keys=["sentiment"]),
        SentimentAccuracyEvaluator(),
    ]
    runner = BenchmarkRunner(provider=provider, evaluators=evaluators)

    prompts = [
        PromptTemplate(
            name="sentiment_v3_cot_json",
            template_str="Review: {customer_review}",
        )
    ]
    cases = [
        TestCase(
            id="c1",
            input_vars={"customer_review": "The product is broken and terrible."},
            expected={"sentiment": "NEGATIVE"},
        )
    ]

    report = runner.run_benchmark(prompts=prompts, test_cases=cases)

    assert len(report.candidates) == 1
    assert report.candidates[0].total_cases == 1
    assert report.candidates[0].pass_rate == 100.0
    assert len(report.detailed_evaluations) == 1
