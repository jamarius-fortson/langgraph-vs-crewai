from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Union

class TaskTier(IntEnum):
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3

class Framework(str, Enum):
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    OPENAI_SDK = "openai-sdk"

@dataclass
class LLMCallRecord:
    request: Dict[str, Any]
    response: Dict[str, Any]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ToolCallRecord:
    tool_name: str
    input: Dict[str, Any]
    output: Any
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ExecutionTrace:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    node_transitions: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class ResourceSample:
    timestamp: datetime
    cpu_percent: float
    memory_mb: float

@dataclass
class RunMeasurement:
    run_id: str
    task_id: str
    framework: Framework
    framework_version: str
    iteration: int
    timestamp: datetime
    
    total_latency_ms: float
    llm_latency_ms: float
    tool_latency_ms: float
    framework_overhead_ms: float
    
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    
    task_success: bool
    task_score: float
    grader_details: Dict[str, Any]
    
    llm_calls_count: int
    tool_calls_count: int
    agent_steps_count: int
    retry_count: int
    
    errors_encountered: int
    errors_recovered: bool
    
    peak_memory_mb: float
    avg_cpu_percent: float
    
    llm_model: str
    python_version: str
    docker_image_hash: str
    
    final_error: Optional[str] = None
    llm_call_log: List[LLMCallRecord] = field(default_factory=list)
    tool_call_log: List[ToolCallRecord] = field(default_factory=list)
    execution_trace: Optional[ExecutionTrace] = None
    resource_timeline: List[ResourceSample] = field(default_factory=list)

@dataclass
class TaskSpec:
    id: str
    name: str
    version: str
    tier: TaskTier
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    system_prompt: str
    tools: List[Dict[str, Any]] = field(default_factory=list)
