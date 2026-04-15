import yaml
from pathlib import Path
from benchmark.cli import app
from benchmark.runner import BenchmarkRunner
from benchmark.registry import TaskRegistry
from frameworks.langgraph.adapter import LangGraphAdapter
from frameworks.crewai.adapter import CrewAIAdapter
from benchmark.types import Framework

def load_config():
    with open("benchmark.yaml", "r") as f:
        return yaml.safe_load(f)

# Patching cli.py with actual Runner logic
# In a real project, this would be in cli.py directly or imported
@app.command()
def run(
    frameworks: str = "langgraph,crewai",
    tasks: str = "all",
    n: int = 5,
    mode: str = "live",
):
    print("Run command invoked")
    config = load_config()
    config["output"]["results_dir"] = "results/temp"
    
    registry = TaskRegistry(Path("tasks/"))
    print(f"Loading tasks from {registry.tasks_dir}")
    registry.load_tasks()
    print(f"Available tasks: {registry.list_tasks()}")
    
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
    print(f"Running tasks: {task_list} across frameworks: {framework_list}")
    
    runner.run_all(framework_list, task_list, n)
    print("Benchmark run completed successfully.")
    
    from benchmark.reporting.markdown import MarkdownReporter
    reporter = MarkdownReporter(Path("results/temp"))
    reporter.generate(Path("results/temp/report.md"))
    print("Report generated: results/temp/report.md")

if __name__ == "__main__":
    app()
