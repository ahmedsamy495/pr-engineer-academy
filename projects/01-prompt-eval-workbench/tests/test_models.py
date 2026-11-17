import pytest
from src.models import PromptTemplate, TestCase, ExecutionResult


def test_prompt_template_render():
    prompt = PromptTemplate(
        name="test_prompt",
        template_str="Hello {name}, your code is {status}!",
    )
    rendered = prompt.render({"name": "Alice", "status": "approved"})
    assert rendered == "Hello Alice, your code is approved!"


def test_prompt_template_missing_variable():
    prompt = PromptTemplate(
        name="test_prompt",
        template_str="Hello {name}, welcome to {place}!",
    )
    with pytest.raises(ValueError) as exc_info:
        prompt.render({"name": "Bob"})
    assert "place" in str(exc_info.value)


def test_test_case_instantiation():
    case = TestCase(
        id="case_01",
        input_vars={"customer_review": "Great product!"},
        expected={"sentiment": "POSITIVE"},
        tags=["unit-test"],
    )
    assert case.id == "case_01"
    assert case.expected["sentiment"] == "POSITIVE"
