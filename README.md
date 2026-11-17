# 🎓 Prompt Engineering Academy (`pr-engineer-academy`)

> **Master Prompt Engineering as a Software Discipline.**  
> Moving beyond conversational "prompt whispering" to empirical, reproducible, and production-ready LLM application engineering.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🌟 Academy Roadmap

The Academy is structured into hands-on, runnable projects that build from core prompt fundamentals to production multi-agent architectures and LLM evaluation suites.

| # | Project | Focus Area | Core Concepts | Status |
|---|:---|:---|:---|:---|
| **01** | [**`prompt-eval-workbench`**](./projects/01-prompt-eval-workbench) | **Empirical Evaluations & A/B Testing** | Zero/Few-Shot, CoT, JSON Schema, LLM-as-a-Judge, Rich CLI | 🚀 **Active** |
| **02** | `rag-retrieval-lab` | RAG & Context Augmentation | Chunking strategies, semantic search, hybrid retrieval, re-ranking | 📅 Planned |
| **03** | `structured-agent-workflows` | ReAct & Function Calling | Tool use, deterministic state machines, error recovery | 📅 Planned |
| **04** | `prompt-security-guardrails` | Defenses & Moderation | Prompt injection prevention, PII redaction, jailbreak evals | 📅 Planned |
| **05** | `model-distillation-tuning` | Efficiency & Cost Optimization | Dataset curation, fine-tuning, small-model task distillation | 📅 Planned |

---

## 🚀 Featured Project: `01-prompt-eval-workbench`

Our foundational project demonstrates how to test and benchmark prompts like unit tests:

- **Empirical A/B Benchmarking**: Compare candidate prompts across multi-case datasets with side-by-side terminal diffs.
- **Multi-Tier Evaluators**:
  - *Deterministic*: Strict JSON Schema validation, regex compliance, latency, token count.
  - *Model-Assisted*: LLM-as-a-Judge scoring with customizable rubrics and reasoning.
- **Offline-First (Mock Provider)**: Runs instantly without API keys or external dependencies.
- **Live LLM Support**: Seamlessly switch to Gemini, OpenAI, or local models.

👉 **[Dive into Project 01](./projects/01-prompt-eval-workbench)**

---

## 🛠️ Repository Quickstart

### Prerequisites
- Python 3.10 or newer (or Node.js / Docker depending on module)
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/pr-engineer-academy.git
cd pr-engineer-academy

# Navigate to Project 01
cd projects/01-prompt-eval-workbench

# Create a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the benchmark CLI with built-in mock mode
python -m src.cli benchmark
```

---

## 📐 Academy Design Principles

1. **Empirical over Intuitive**: Never deploy a prompt without a benchmark dataset and quantifiable metrics.
2. **Deterministic Outputs**: Enforce structured outputs (JSON Schemas, typed Pydantic models) over conversational prose.
3. **Modular & Model-Agnostic**: Core prompt templates and evaluators remain decoupled from specific model providers.
4. **Test-Driven**: Every project ships with an automated test suite (`pytest`) and sample datasets.

---

## 📄 License
This repository is open-sourced under the [MIT License](LICENSE).
