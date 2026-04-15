from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from benchmark.types import (
    TaskSpec, 
    RunMeasurement, 
    Framework, 
    ExecutionTrace
)
from benchmark.llm_proxy import LLMInterceptor

class FrameworkAdapter(ABC):
    """Base adapter that every framework implementation must extend."""

    @abstractmethod
    def name(self) -> Framework:
        """Canonical framework name."""
        pass

    @abstractmethod
    def version(self) -> str:
        """Exact installed framework version."""
        pass

    @abstractmethod
    def setup(self, config: Dict[str, Any]) -> None:
        """One-time setup: install deps, configure LLM client, etc."""
        pass

    @abstractmethod
    def execute_task(self, task: TaskSpec, interceptor: LLMInterceptor) -> RunMeasurement:
        """
        Run a single task and return structured result.
        
        This is the core method. The implementation must:
        1. Build the agent/crew/graph using framework idioms
        2. Execute the task with the provided inputs
        3. Capture all LLM calls via the interceptor
        4. Return the result with output + metadata
        """
        pass

    @abstractmethod
    def get_execution_trace(self) -> ExecutionTrace:
        """
        Return the framework-specific execution trace.
        
        Includes: steps taken, tools called, agent transitions, 
        internal retries, memory operations.
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup: close connections, free resources."""
        pass

    def supports_feature(self, feature: str) -> bool:
        """Declare whether this framework natively supports a feature."""
        return False
