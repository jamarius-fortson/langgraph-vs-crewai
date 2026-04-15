import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from benchmark.types import LLMCallRecord
from benchmark.measurement import TokenCounter

class LLMInterceptor:
    """Intercepts and records LLM API calls for measurement integrity."""

    def __init__(self, model_version: str):
        self.model_version = model_version
        self.token_counter = TokenCounter(model_version)
        self.calls: List[LLMCallRecord] = []

    def record_call(self, request: Dict[str, Any], response: Dict[str, Any], duration_ms: float) -> LLMCallRecord:
        """Process and record a single LLM API transaction."""
        
        # Calculate tokens independently (TikToken)
        input_text = self._extract_text(request.get("messages", []))
        output_text = self._extract_response_text(response)
        
        input_tokens = self.token_counter.count_tokens(input_text)
        output_tokens = self.token_counter.count_tokens(output_text)

        record = LLMCallRecord(
            request=request,
            response=response,
            latency_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model_version,
            timestamp=datetime.utcnow()
        )
        self.calls.append(record)
        return record

    def _extract_text(self, messages: List[Dict[str, Any]]) -> str:
        """Concatenate all message content and tool structure for token counting."""
        parts = []
        for m in messages:
            # Content
            if "content" in m and m["content"]:
                parts.append(m["content"])
            
            # Tool Calls (Assistant Message)
            if "tool_calls" in m and m["tool_calls"]:
                for tc in m["tool_calls"]:
                    f = tc.get("function", {})
                    parts.append(f.get("name", ""))
                    parts.append(f.get("arguments", ""))
            
            # Tool Result (Tool Message)
            if "name" in m:
                parts.append(m["name"])
                
        return "\n".join(parts)

    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """Extract completion text and tool calls from API response."""
        try:
            choice = response["choices"][0]["message"]
            content = choice.get("content") or ""
            
            # Include tool call data in the token count
            tool_parts = []
            if "tool_calls" in choice and choice["tool_calls"]:
                for tc in choice["tool_calls"]:
                    f = tc.get("function", {})
                    tool_parts.append(f.get("name", ""))
                    tool_parts.append(f.get("arguments", ""))
            
            return content + "\n".join(tool_parts)
        except (KeyError, IndexError, TypeError):
            return ""

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate metrics across all intercepted calls."""
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": sum(c.input_tokens for c in self.calls),
            "total_output_tokens": sum(c.output_tokens for c in self.calls),
            "total_latency_ms": sum(c.latency_ms for c in self.calls)
        }
