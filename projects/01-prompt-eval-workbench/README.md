# 🔬 Project 01: Prompt Evaluation & Benchmarking Workbench

> **Track:** Systematic Prompt Engineering & LLM App Development  
> **Skill Level:** Intermediate  
> **Tech Stack:** Python 3.10+, Pydantic v2, Typer, Rich, Pytest

---

## 🎯 What You Will Learn

1. **Why Intuition Fails in Production:** Why tweaking prompts ad-hoc leads to subtle regressions, and how empirical test suites solve this.
2. **Core Prompting Paradigms:**
   - **Zero-Shot (`v1_zeroshot`):** Raw task statement without examples or schema guardrails. High risk of format divergence.
   - **Few-Shot In-Context Learning (`v2_fewshot`):** Providing demonstration pairs to stabilize format and style.
   - **Chain-of-Thought with Schema Enforcement (`v3_cot_json`):** Directing the model to reason before outputting strictly structured JSON.
3. **Multi-Tier Evaluators:**
   - **Deterministic:** Strict JSON schema parsing, key validation, regex matching, latency budgets.
   - **Model-Assisted (LLM-as-a-Judge):** Rubric-based qualitative scoring (1–5 scale).
4. **Offline Mock Architecture:** Building deterministic mocks so LLM pipelines can be tested in CI/CD without API tokens or network latency.

---

## 📂 Project Structure

```
01-prompt-eval-workbench/
├── config.example.env         # API credentials template
├── datasets/
│   └── customer_feedback.json # Test cases with inputs and ground truth
├── prompts/                   # Versioned candidate prompt files
│   ├── sentiment_v1_zeroshot.txt
│   ├── sentiment_v2_fewshot.txt
│   └── sentiment_v3_cot_json.txt
├── requirements.txt           # Dependencies
├── pyproject.toml             # Build configuration
├── src/
│   ├── cli.py                 # Rich CLI terminal application
│   ├── models.py              # Pydantic data models
│   ├── runner.py              # Benchmarking engine
│   ├── providers/             # LLM connectors (Mock, Gemini, base)
│   └── evaluators/            # Deterministic and LLM Judge metrics
└── tests/                     # Pytest suite
```

---

## 🚀 Quickstart Guide

### 1. Setup Environment
```bash
# Navigate to project directory
cd projects/01-prompt-eval-workbench

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Inspect Available Prompts
```bash
python -m src.cli list-prompts
```

### 3. Run Benchmark (Offline Mock Mode)
Runs out of the box with zero API keys required:
```bash
python -m src.cli benchmark
```

You will see a formatted comparison table evaluating all candidates side-by-side:
- **Pass Rate %**
- **Average Latency (ms)**
- **JSON Schema Validity %**
- **Classification Accuracy %**
- **LLM-as-a-Judge Rubric Score (1–5)**

### 4. Run Benchmark with Live Gemini (Optional)
To test against Google Gemini models:
1. Copy `config.example.env` to `.env` and set `GEMINI_API_KEY`.
2. Run:
```bash
python -m src.cli benchmark --live
```

---

## 🧪 Running Automated Tests

Run the test suite using `pytest`:
```bash
pytest -v
```

---

## 🧩 Extending the Project

- **Add a new prompt:** Create `prompts/sentiment_v4_system_role.txt`. The CLI will automatically discover and benchmark it.
- **Add custom test cases:** Append objects to `datasets/customer_feedback.json`.
- **Add a new evaluator:** Subclass `BaseEvaluator` in `src/evaluators/` and register it in `cli.py`.
