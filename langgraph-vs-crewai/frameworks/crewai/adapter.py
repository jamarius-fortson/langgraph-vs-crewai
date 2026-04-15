from typing import Any, Dict
from benchmark.types import Framework, TaskSpec, RunMeasurement, ExecutionTrace
from frameworks.base import FrameworkAdapter
from benchmark.llm_proxy import LLMInterceptor
from datetime import datetime

class CrewAIAdapter(FrameworkAdapter):
    def name(self) -> Framework:
        return Framework.CREWAI

    def version(self) -> str:
        return "0.83.0"

    def setup(self, config: Dict[str, Any]) -> None:
        print("CrewAIAdapter: Setting up...")

    def execute_task(self, task: TaskSpec, interceptor: LLMInterceptor) -> RunMeasurement:
        print(f"CrewAIAdapter: Executing task {task.id}...")
        
        # Simulate an LLM call through the interceptor
        interceptor.record_call(
            request={"messages": [{"role": "system", "content": "You are a helpful agent."}, {"role": "user", "content": "Research AI trends."}]},
            response={"choices": [{"message": {"content": "Here are the trends..."}}]},
            duration_ms=1200.0
        )
        interceptor.record_call(
            request={"messages": [{"role": "system", "content": "Aggregate the results."}, {"role": "user", "content": "Summarize trends."}]},
            response={"choices": [{"message": {"content": "AI is growing fast."}}]},
            duration_ms=800.0
        )
        
        # Simulate execution for now
        return RunMeasurement(
            run_id="",
            task_id=task.id,
            framework=self.name(),
            framework_version=self.version(),
            iteration=0,
            timestamp=datetime.utcnow(),
            total_latency_ms=0,
            llm_latency_ms=0, # Will be filled by runner
            tool_latency_ms=200,
            framework_overhead_ms=200,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            total_cost_usd=0,
            task_success=True,
            task_score=1.0,
            grader_details={"correct_answer": 1.0},
            llm_calls_count=0,
            tool_calls_count=1,
            agent_steps_count=5,
            retry_count=0,
            errors_encountered=0,
            errors_recovered=True,
            peak_memory_mb=256,
            avg_cpu_percent=3.5,
            llm_model="gpt-4o-2024-11-20",
            python_version="3.11",
            docker_image_hash="sha256:def"
        )

    def get_execution_trace(self) -> ExecutionTrace:
        return ExecutionTrace()

    def teardown(self) -> None:
        print("CrewAIAdapter: Tearing down...")
