from abc import ABC, abstractmethod
from typing import Optional
from src.models import ExecutionResult


class BaseLLMProvider(ABC):
    """Abstract interface for LLM backends (Mock, Gemini, OpenAI, etc.)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        prompt_name: str = "default",
        test_case_id: str = "default",
    ) -> ExecutionResult:
        """Execute the prompt and return an ExecutionResult."""
        pass
