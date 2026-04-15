import json
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Agent Framework Benchmark Suite CLI")
console = Console()

from pathlib import Path
import yaml
from benchmark.runner import BenchmarkRunner
from benchmark.registry import TaskRegistry
from frameworks.langgraph.adapter import LangGraphAdapter
from frameworks.crewai.adapter import CrewAIAdapter
from benchmark.types import Framework

def load_config():
    with open("benchmark.yaml", "r") as f:
        return yaml.safe_load(f)

@app.command()
def run(
    frameworks: str = typer.Option("langgraph,crewai", help="Comma-separated list of frameworks"),
    tasks: str = typer.Option("all", help="Comma-separated list of tasks or 'all'"),
    n: int = typer.Option(5, help="Number of iterations per task"),
    mode: str = typer.Option("live", help="Execution mode: live, replay, dry-run"),
    output: Path = typer.Option(Path("results/"), help="Directory for results output"),
):
    """Run the benchmark suite."""
    console.print(f"[bold green]Starting benchmark run...[/bold green]")
    
    config = load_config()
    config["output"]["results_dir"] = str(output)
    
    registry = TaskRegistry(Path("tasks/"))
    registry.load_tasks()
    
    # Initialize real adapters (will still use mock logic if libraries aren't installed)
    adapters = {
        Framework.LANGGRAPH: LangGraphAdapter(),
        Framework.CREWAI: CrewAIAdapter()
    }
    
    runner = BenchmarkRunner(registry, adapters, config)
    
    framework_list = frameworks.split(",")
    if tasks == "all":
        task_list = registry.list_tasks()
    else:
        task_list = tasks.split(",")
    
    runner.run_all(framework_list, task_list, n)
    console.print("[bold cyan]Benchmark run completed successfully.[/bold cyan]")

    # Auto-generate report
    from benchmark.reporting.markdown import MarkdownReporter
    reporter = MarkdownReporter(output)
    report_file = output / "report.md"
    reporter.generate(report_file)
    console.print(f"Report generated: [bold white]{report_file}[/bold white]")

@app.command()
def list_tasks():
    """List all available benchmark tasks."""
    # TODO: Implement listing logic
    console.print("Task list coming soon...")

if __name__ == "__main__":
    app()
