from datetime import datetime, timezone
from typing import List, Dict
from src.models import (
    PromptTemplate,
    TestCase,
    CaseEvaluation,
    BenchmarkSummary,
    BenchmarkReport,
)
from src.providers.base import BaseLLMProvider
from src.evaluators.base import BaseEvaluator


class BenchmarkRunner:
    """Orchestrates executing candidate prompts across test datasets and computing evals."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        evaluators: List[BaseEvaluator],
    ):
        self.provider = provider
        self.evaluators = evaluators

    def run_benchmark(
        self,
        prompts: List[PromptTemplate],
        test_cases: List[TestCase],
    ) -> BenchmarkReport:
        all_case_evals: List[CaseEvaluation] = []
        candidate_summaries: List[BenchmarkSummary] = []

        for prompt in prompts:
            prompt_evals: List[CaseEvaluation] = []
            metric_totals: Dict[str, float] = {e.name: 0.0 for e in self.evaluators}
            passed_cases_count = 0
            total_latency = 0.0

            for case in test_cases:
                # 1. Render prompt with variables
                rendered = prompt.render(case.input_vars)

                # 2. Query provider
                execution = self.provider.generate(
                    prompt=rendered,
                    system_instruction=prompt.system_instruction,
                    prompt_name=prompt.name,
                    test_case_id=case.id,
                )
                total_latency += execution.latency_ms

                # 3. Run all evaluators
                case_metrics = []
                case_passed = True

                for evaluator in self.evaluators:
                    metric_res = evaluator.evaluate(execution, case)
                    case_metrics.append(metric_res)
                    metric_totals[evaluator.name] += metric_res.score
                    if not metric_res.passed:
                        case_passed = False

                if case_passed:
                    passed_cases_count += 1

                case_eval = CaseEvaluation(
                    test_case_id=case.id,
                    prompt_name=prompt.name,
                    execution=execution,
                    metrics=case_metrics,
                    all_passed=case_passed,
                )
                prompt_evals.append(case_eval)
                all_case_evals.append(case_eval)

            # Summarize metrics for this prompt candidate
            n_cases = max(1, len(test_cases))
            metric_averages = {
                name: round(total / n_cases, 2)
                for name, total in metric_totals.items()
            }
            pass_rate = round((passed_cases_count / n_cases) * 100, 1)
            avg_latency = round(total_latency / n_cases, 1)

            summary = BenchmarkSummary(
                prompt_name=prompt.name,
                total_cases=len(test_cases),
                passed_cases=passed_cases_count,
                pass_rate=pass_rate,
                avg_latency_ms=avg_latency,
                metric_averages=metric_averages,
            )
            candidate_summaries.append(summary)

        return BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            candidates=candidate_summaries,
            detailed_evaluations=all_case_evals,
        )
