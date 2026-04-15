from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
from dataclasses import dataclass

@dataclass
class GradeCriterion:
    name: str
    score: float
    weight: float
    details: str

@dataclass
class GradeResult:
    score: float
    passed: bool
    criteria: List[GradeCriterion]
    explanation: str

class TaskGrader(ABC):
    """Base grader for benchmark tasks."""

    @abstractmethod
    def grade(self, task_output: Dict[str, Any], expected: Dict[str, Any], 
              execution_trace: Any) -> GradeResult:
        pass

class StandardGrader(TaskGrader):
    """Standard automated grader supporting various rubric types."""
    
    def grade(self, task_output: Dict[str, Any], expected: Dict[str, Any], 
              execution_trace: Any) -> GradeResult:
        criteria = []
        total_score = 0.0
        
        # Exact Numeric Match
        if "answer" in expected:
            expected_val = expected["answer"]
            actual_val = task_output.get("answer")
            
            if actual_val is not None:
                try:
                    score = 1.0 if abs(float(actual_val) - float(expected_val)) < 0.001 else 0.0
                except (ValueError, TypeError):
                    score = 0.0
                
                criteria.append(GradeCriterion(
                    name="correct_answer",
                    score=score,
                    weight=0.7,
                    details=f"Expected {expected_val}, got {actual_val}"
                ))
                total_score += score * 0.7

        # Placeholder for other grading logic
        # ...
        
        return GradeResult(
            score=total_score,
            passed=total_score >= 0.7,
            criteria=criteria,
            explanation="Automated grading complete."
        )
