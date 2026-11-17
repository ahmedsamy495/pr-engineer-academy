import os
import time
from typing import Optional
from src.providers.base import BaseLLMProvider
from src.models import ExecutionResult


class GeminiProvider(BaseLLMProvider):
    """
    Live LLM provider connecting to Google Gemini API via google-genai SDK.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GeminiProvider requires GEMINI_API_KEY environment variable or explicit api_key."
            )
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "google-genai package is required for GeminiProvider. Run `pip install google-genai`."
            )

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        prompt_name: str = "default",
        test_case_id: str = "default",
    ) -> ExecutionResult:
        start_time = time.perf_counter()
        
        config = {}
        if system_instruction:
            config["system_instruction"] = system_instruction

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            raw_text = response.text or ""
            latency_ms = (time.perf_counter() - start_time) * 1000
            token_estimate = len(raw_text.split()) * 2

            return ExecutionResult(
                prompt_name=prompt_name,
                test_case_id=test_case_id,
                rendered_prompt=prompt,
                raw_response=raw_text,
                latency_ms=round(latency_ms, 2),
                token_estimate=token_estimate,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                prompt_name=prompt_name,
                test_case_id=test_case_id,
                rendered_prompt=prompt,
                raw_response="",
                latency_ms=round(latency_ms, 2),
                token_estimate=0,
                error=str(e),
            )
