from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """Represents a versioned prompt candidate."""
    name: str = Field(..., description="Unique identifier for the prompt version, e.g. sentiment_v1_zeroshot")
    version: str = Field(default="1.0.0", description="Semantic version string")
    template_str: str = Field(..., description="The raw prompt string with {variable} placeholders")
    description: str = Field(default="", description="Description of the prompt strategy (e.g., Few-shot, CoT)")
    system_instruction: Optional[str] = Field(default=None, description="Optional system instruction or role")

    def render(self, variables: Dict[str, Any]) -> str:
        """Render the prompt string with provided variable substitutions."""
        try:
            return self.template_str.format(**variables)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(f"Prompt '{self.name}' requires variable '{missing_key}', but it was not provided.") from e


class TestCase(BaseModel):
    """Represents a single input case for benchmarking."""
    __test__ = False  # Prevent pytest from treating this Pydantic model as a test suite
    id: str = Field(..., description="Unique test case identifier")
    input_vars: Dict[str, Any] = Field(..., description="Variable inputs injected into the prompt template")
    expected: Optional[Dict[str, Any]] = Field(default=None, description="Ground truth or target expectations")
    tags: List[str] = Field(default_factory=list, description="Categorization tags (e.g., edge-case, negative)")


class ExecutionResult(BaseModel):
    """Raw execution result of a prompt against an LLM provider."""
    prompt_name: str
    test_case_id: str
    rendered_prompt: str
    raw_response: str
    latency_ms: float
    token_estimate: int
    error: Optional[str] = None


class EvalMetricResult(BaseModel):
    """Result of an individual evaluator check."""
    metric_name: str
    score: float = Field(..., description="Normalized score, usually 0.0 to 1.0 (or 1-5 for LLM judge)")
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = Field(default=None, description="Explanation or rationale (especially from LLM judge)")


class CaseEvaluation(BaseModel):
    """Aggregated evaluation for a single test case."""
    test_case_id: str
    prompt_name: str
    execution: ExecutionResult
    metrics: List[EvalMetricResult]
    all_passed: bool


class BenchmarkSummary(BaseModel):
    """Aggregated performance report for a single prompt candidate."""
    prompt_name: str
    total_cases: int
    passed_cases: int
    pass_rate: float
    avg_latency_ms: float
    metric_averages: Dict[str, float]


class BenchmarkReport(BaseModel):
    """Final comparative report of all prompt candidates."""
    timestamp: str
    candidates: List[BenchmarkSummary]
    detailed_evaluations: List[CaseEvaluation]
