import time
from typing import Dict, Any, List
import tiktoken
from benchmark.types import LLMCallRecord, ToolCallRecord

class TokenCounter:
    """External token counting using tiktoken for fairness."""
    
    def __init__(self, model: str = "gpt-4o"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Rough estimate of tokens in a list of messages."""
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += self.count_tokens(value)
                if key == "name":
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

class LatencyTimer:
    """Nanosecond precision timer for latency measurement."""
    
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = time.perf_counter_ns()

    def stop(self):
        self.end_time = time.perf_counter_ns()

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) / 1_000_000.0

def calculate_cost(input_tokens: int, output_tokens: int, pricing: Dict[str, float]) -> float:
    """Calculate cost based on token counts and pricing model."""
    input_cost = (input_tokens / 1000) * pricing.get("input_per_1k", 0)
    output_cost = (output_tokens / 1000) * pricing.get("output_per_1k", 0)
    return input_cost + output_cost
