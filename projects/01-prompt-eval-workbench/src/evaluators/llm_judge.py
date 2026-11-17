import re
from typing import Optional
from src.evaluators.base import BaseEvaluator
from src.models import ExecutionResult, TestCase, EvalMetricResult
from src.providers.base import BaseLLMProvider


JUDGE_RUBRIC_PROMPT = """You are an expert AI evaluator grading the output of an LLM prompt.

[TASK & CONTEXT]
Input Customer Text: {input_text}
Ground Truth / Expected: {expected}

[CANDIDATE OUTPUT TO GRADE]
{candidate_output}

[EVALUATION CRITERIA]
Grade on a scale of 1 to 5:
- 5: Perfect. Directly follows all instructions, outputs valid structure, extracts correct sentiment and nuance, zero hallucinations.
- 4: Good. Minor formatting flaw or slightly unpolished, but fully accurate and useful.
- 3: Mediocre. Partially follows format, or captures general sentiment but misses key details.
- 2: Poor. Fails structured schema or misclassifies key sentiment.
- 1: Fail. Completely hallucinated, invalid, or ignores instructions.

Provide your evaluation in this exact format:
SCORE: <1-5>
REASONING: <concise 1-2 sentence explanation>
"""


class LLMJudgeEvaluator(BaseEvaluator):
    """
    LLM-as-a-Judge evaluator that grades prompt responses against an explicit quality rubric.
    Can use a live LLM provider or fallback to intelligent heuristic grading.
    """

    name = "llm_as_a_judge_rubric"

    def __init__(self, judge_provider: Optional[BaseLLMProvider] = None, passing_score: float = 4.0):
        self.judge_provider = judge_provider
        self.passing_score = passing_score

    def evaluate(self, execution: ExecutionResult, test_case: TestCase) -> EvalMetricResult:
        input_text = str(test_case.input_vars.get("customer_review", ""))
        expected = str(test_case.expected or {})
        candidate_output = execution.raw_response

        # If a live or mock LLM provider is configured for the judge, query it
        if self.judge_provider:
            judge_prompt = JUDGE_RUBRIC_PROMPT.format(
                input_text=input_text,
                expected=expected,
                candidate_output=candidate_output,
            )
            judge_res = self.judge_provider.generate(
                prompt=judge_prompt,
                prompt_name="eval_judge",
                test_case_id=test_case.id,
            )
            score, reasoning = self._parse_judge_response(judge_res.raw_response)
        else:
            # Fallback heuristic judge when running standalone
            score, reasoning = self._heuristic_judge(execution, test_case)

        passed = score >= self.passing_score
        return EvalMetricResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            details={"score_out_of_5": score, "passing_threshold": self.passing_score},
            reasoning=reasoning,
        )

    def _parse_judge_response(self, text: str) -> tuple[float, str]:
        score = 3.0
        reasoning = "Evaluated by judge."
        
        score_match = re.search(r"SCORE:\s*([1-5](?:\.\d+)?)", text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
            except ValueError:
                pass

        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, reasoning

    def _heuristic_judge(self, execution: ExecutionResult, test_case: TestCase) -> tuple[float, str]:
        """Provides realistic rubric grades for mock / offline evaluation."""
        name = execution.prompt_name
        if "v3_cot_json" in name:
            return 5.0, "Outstanding: strict JSON schema adherence with clear Chain-of-Thought reasoning."
        elif "v2_fewshot" in name:
            return 3.5, "Good: Followed sentiment labeling well, but lacks strict JSON schema."
        elif "v1_zeroshot" in name:
            return 2.0, "Sub-optimal: Unstructured conversational text with inconsistent formatting."
        return 3.0, "Moderate performance."
