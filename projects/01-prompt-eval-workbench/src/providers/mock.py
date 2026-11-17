import time
import json
from typing import Optional
from src.providers.base import BaseLLMProvider
from src.models import ExecutionResult


class MockProvider(BaseLLMProvider):
    """
    Offline mock provider that simulates realistic LLM behavior.
    Demonstrates differences in output fidelity between Zero-Shot, Few-Shot, and CoT JSON prompts.
    """

    def __init__(self, latency_ms: float = 85.0):
        self.latency_ms = latency_ms

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        prompt_name: str = "default",
        test_case_id: str = "default",
    ) -> ExecutionResult:
        start_time = time.perf_counter()
        
        # Simulate slight network jitter
        simulated_latency = self.latency_ms + (hash(prompt) % 25)

        lower_prompt = prompt.lower()
        is_negative = "terrible" in lower_prompt or "broken" in lower_prompt or "worst" in lower_prompt or "unacceptable" in lower_prompt
        is_neutral = "average" in lower_prompt or "okay" in lower_prompt or "fine" in lower_prompt

        # Tailor simulated response based on the prompt candidate strategy
        if "v1_zeroshot" in prompt_name:
            # Zero-shot often returns conversational filler and inconsistent format
            if is_negative:
                raw_response = "I think the customer is really unhappy. Sentiment: Negative. They had a bad delivery experience."
            elif is_neutral:
                raw_response = "The sentiment seems Neutral or mixed."
            else:
                raw_response = "Sure! The sentiment is Positive. Great review!"

        elif "v2_fewshot" in prompt_name:
            # Few-shot follows structure better, but might miss schema types or strict JSON
            sentiment = "NEGATIVE" if is_negative else ("NEUTRAL" if is_neutral else "POSITIVE")
            raw_response = f"Sentiment: {sentiment}\nCategory: Delivery\nConfidence: 0.9"

        elif "v3_cot_json" in prompt_name:
            # CoT + JSON schema output produces clean, parseable JSON with rationale
            sentiment = "NEGATIVE" if is_negative else ("NEUTRAL" if is_neutral else "POSITIVE")
            score = 0.95 if is_negative else (0.75 if is_neutral else 0.98)
            payload = {
                "thought_process": "Analyzed customer feedback sentiment cues and extracted entities.",
                "sentiment": sentiment,
                "confidence_score": score,
                "urgency_level": "HIGH" if is_negative else "LOW",
                "key_issues": ["shipping delay", "damaged package"] if is_negative else ["general feedback"],
                "action_required": is_negative,
            }
            raw_response = json.dumps(payload, indent=2)

        else:
            # Generic response
            raw_response = json.dumps({
                "sentiment": "NEUTRAL",
                "confidence_score": 0.8,
                "note": "Generic mock output"
            })

        elapsed_ms = (time.perf_counter() - start_time) * 1000 + simulated_latency
        token_estimate = max(10, len(raw_response.split()) * 2)

        return ExecutionResult(
            prompt_name=prompt_name,
            test_case_id=test_case_id,
            rendered_prompt=prompt,
            raw_response=raw_response,
            latency_ms=round(elapsed_ms, 2),
            token_estimate=token_estimate,
        )
