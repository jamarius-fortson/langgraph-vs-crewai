import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class CostCalculator:
    """Calculates LLM costs based on versioned pricing models."""

    def __init__(self, pricing_file: Path):
        with open(pricing_file, "r") as f:
            self.pricing_data = yaml.safe_load(f)

    def calculate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate total USD cost for a given token usage."""
        model_pricing = self.pricing_data.get("models", {}).get(model)
        if not model_pricing:
            # Fallback or alert if model not in pricing model
            return 0.0

        input_rate = model_pricing.get("input_per_1k", 0)
        output_rate = model_pricing.get("output_per_1k", 0)
        
        cost = (input_tokens / 1000.0) * input_rate + (output_tokens / 1000.0) * output_rate
        return cost
