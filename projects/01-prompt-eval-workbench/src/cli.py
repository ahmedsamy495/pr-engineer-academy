import os
import json
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.models import PromptTemplate, TestCase, BenchmarkReport
from src.providers.mock import MockProvider
from src.providers.gemini import GeminiProvider
from src.evaluators.deterministic import (
    JsonSchemaEvaluator,
    SentimentAccuracyEvaluator,
    LatencyBudgetEvaluator,
)
from src.evaluators.llm_judge import LLMJudgeEvaluator
from src.runner import BenchmarkRunner

app = typer.Typer(
    help="Prompt Evaluation & Benchmarking Workbench CLI",
    add_completion=False,
)
console = Console()

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
DATASETS_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports"


def load_prompts() -> List[PromptTemplate]:
    """Discover and load prompt template files."""
    prompts = []
    for path in sorted(PROMPTS_DIR.glob("*.txt")):
        name = path.stem
        content = path.read_text(encoding="utf-8")
        prompts.append(
            PromptTemplate(
                name=name,
                template_str=content,
                description=f"Prompt template loaded from {path.name}",
            )
        )
    return prompts


def load_dataset(dataset_name: str = "customer_feedback.json") -> List[TestCase]:
    """Load test cases from the datasets directory."""
    path = DATASETS_DIR / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    return [TestCase(**item) for item in raw_data]


@app.command()
def list_prompts():
    """List all available prompt candidates in prompts/ directory."""
    prompts = load_prompts()
    table = Table(title="Available Prompt Candidates", box=box.ROUNDED)
    table.add_column("Prompt Identifier", style="cyan bold")
    table.add_column("File Source", style="dim")
    table.add_column("Length (chars)", justify="right")

    for p in prompts:
        table.add_row(p.name, f"prompts/{p.name}.txt", str(len(p.template_str)))

    console.print(table)


@app.command()
def benchmark(
    live: bool = typer.Option(
        False, "--live", "-l", help="Use live Gemini API instead of offline MockProvider"
    ),
    dataset: str = typer.Option(
        "customer_feedback.json", "--dataset", "-d", help="Dataset filename"
    ),
    save_report: bool = typer.Option(
        True, "--save/--no-save", help="Save JSON benchmark report to reports/"
    ),
):
    """Run empirical A/B evaluation benchmark across all prompt candidates."""
    console.print(
        Panel(
            "[bold green]Prompt Evaluation & Benchmarking Workbench[/bold green]\n"
            f"[dim]Mode: {'Live Gemini API' if live else 'Offline Mock Provider (Zero API Keys)'}[/dim]",
            border_style="green",
        )
    )

    prompts = load_prompts()
    if not prompts:
        console.print("[red]No prompt templates found in prompts/ folder![/red]")
        raise typer.Exit(1)

    test_cases = load_dataset(dataset)
    console.print(f"Loaded [cyan]{len(prompts)}[/cyan] prompt candidates and [cyan]{len(test_cases)}[/cyan] test cases.\n")

    # Select provider
    if live:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print("[red]Error: GEMINI_API_KEY is not set. Set it or run without --live.[/red]")
            raise typer.Exit(1)
        provider = GeminiProvider(api_key=api_key)
    else:
        provider = MockProvider()

    # Configure Evaluators
    evaluators = [
        JsonSchemaEvaluator(required_keys=["sentiment", "confidence_score"]),
        SentimentAccuracyEvaluator(),
        LatencyBudgetEvaluator(max_latency_ms=1000.0),
        LLMJudgeEvaluator(),
    ]

    runner = BenchmarkRunner(provider=provider, evaluators=evaluators)
    with console.status("[bold blue]Executing benchmark evaluation matrix...[/bold blue]"):
        report = runner.run_benchmark(prompts=prompts, test_cases=test_cases)

    # Render results table
    table = Table(
        title="Benchmark Comparative Results (Prompt A/B Matrix)",
        box=box.DOUBLE_EDGE,
        header_style="bold magenta",
    )
    table.add_column("Prompt Candidate", style="bold cyan")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("JSON Schema", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Judge Score (1-5)", justify="right")

    for c in report.candidates:
        pass_color = "green" if c.pass_rate >= 80 else ("yellow" if c.pass_rate >= 50 else "red")
        table.add_row(
            c.prompt_name,
            f"[{pass_color}]{c.pass_rate}%[/{pass_color}]",
            f"{c.avg_latency_ms} ms",
            f"{int(c.metric_averages.get('json_schema_validity', 0) * 100)}%",
            f"{int(c.metric_averages.get('sentiment_accuracy', 0) * 100)}%",
            f"{c.metric_averages.get('llm_as_a_judge_rubric', 0):.1f} / 5.0",
        )

    console.print(table)

    # Save report
    if save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_file = REPORTS_DIR / f"benchmark_{report.timestamp.replace(':', '-').replace('.', '-')}.json"
        report_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"\n[green]Saved full benchmark report to:[/green] [dim]{report_file}[/dim]")


if __name__ == "__main__":
    app()
