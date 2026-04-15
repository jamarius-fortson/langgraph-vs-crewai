import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Type
import json
from benchmark.types import (
    TaskSpec, 
    Framework, 
    RunMeasurement, 
    LLMCallRecord, 
    ToolCallRecord
)
from benchmark.registry import TaskRegistry
from benchmark.measurement import LatencyTimer, TokenCounter
from benchmark.llm_proxy import LLMInterceptor
from benchmark.cost_calculator import CostCalculator
from benchmark.docker_manager import DockerManager
from benchmark.results_db import ResultsDatabase
from frameworks.base import FrameworkAdapter

class BenchmarkRunner:
    """Orchestrates the benchmark execution across frameworks and tasks."""

    def __init__(
        self, 
        registry: TaskRegistry, 
        adapters: Dict[Framework, FrameworkAdapter],
        config: Dict[str, Any]
    ):
        self.registry = registry
        self.adapters = adapters
        self.config = config
        self.results_dir = Path(config["output"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize measurement tools
        pricing_path = Path("pricing") / f"{config['measurement']['cost_model']}.yaml"
        self.cost_calculator = CostCalculator(pricing_path)
        self.docker_manager = DockerManager(config)
        
        # Initialize SQLite database for scalability
        self.db = ResultsDatabase(self.results_dir / "benchmarks.sqlite")

    def run_all(self, framework_names: List[str], task_ids: List[str], n_iterations: int):
        """Run multiple iterations for each task across frameworks."""
        for task_id in task_ids:
            task = self.registry.get_task(task_id)
            if not task:
                print(f"Task {task_id} not found.")
                continue

            for framework_name in framework_names:
                framework = Framework(framework_name)
                adapter = self.adapters.get(framework)
                if not adapter:
                    print(f"Adapter for {framework_name} not found.")
                    continue

                for i in range(n_iterations):
                    print(f"Running task {task_id} with framework {framework_name} [Iteration {i+1}/{n_iterations}]")
                    result = self.run_iteration(adapter, task, i + 1)
                    self.save_result(result)

    def run_iteration(self, adapter: FrameworkAdapter, task: TaskSpec, iteration: int) -> RunMeasurement:
        """Run a single iteration of a task for a framework."""
        run_id = str(uuid.uuid4())
        timer = LatencyTimer()
        interceptor = LLMInterceptor(model_version=self.config["llm"]["model"])
        
        # Setup framework
        adapter.setup(self.config)
        
        timer.start()
        try:
            # Execute task via adapter
            result = adapter.execute_task(task, interceptor)
            timer.stop()
            
            # Fill in measurements from interceptor
            summary = interceptor.get_summary()
            
            result.run_id = run_id
            result.iteration = iteration
            result.total_latency_ms = timer.duration_ms
            result.llm_latency_ms = summary["total_latency_ms"]
            result.total_input_tokens = summary["total_input_tokens"]
            result.total_output_tokens = summary["total_output_tokens"]
            result.total_tokens = result.total_input_tokens + result.total_output_tokens
            result.llm_calls_count = summary["total_calls"]
            result.llm_call_log = interceptor.calls
            
            # Calculate cost
            result.total_cost_usd = self.cost_calculator.calculate(
                result.llm_model, 
                result.total_input_tokens, 
                result.total_output_tokens
            )
            
            return result
        except Exception as e:
            timer.stop()
            print(f"Execution failed: {e}")
            # Return a failed result
            return RunMeasurement(
                run_id=run_id,
                task_id=task.id,
                framework=adapter.name(),
                framework_version=adapter.version(),
                iteration=iteration,
                timestamp=datetime.utcnow(),
                total_latency_ms=timer.duration_ms,
                llm_latency_ms=0,
                tool_latency_ms=0,
                framework_overhead_ms=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_cost_usd=0,
                task_success=False,
                task_score=0.0,
                grader_details={},
                llm_calls_count=0,
                tool_calls_count=0,
                agent_steps_count=0,
                retry_count=0,
                errors_encountered=1,
                errors_recovered=False,
                final_error=str(e),
                peak_memory_mb=0,
                avg_cpu_percent=0,
                llm_model=self.config["llm"]["model"],
                python_version="3.11",
                docker_image_hash="sha256:unknown"
            )
        finally:
            adapter.teardown()

    def save_result(self, result: RunMeasurement):
        """Save measurement to both JSON and SQLite."""
        # 1. Save to SQLite (Scalable for N > 100)
        self.db.save_run(result)
        
        # 2. Save individual JSON for inspection/replay
        file_path = self.results_dir / f"{result.task_id}_{result.framework.value}_{result.run_id}.json"
        
        def serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        with open(file_path, "w") as f:
            # Convert to dict (primitive way for now)
            data = result.__dict__.copy()
            data["framework"] = result.framework.value
            data["timestamp"] = result.timestamp.isoformat()
            
            # Use a custom json.dumps to handle nested objects and datetimes
            json.dump(data, f, indent=2, default=serializer)
