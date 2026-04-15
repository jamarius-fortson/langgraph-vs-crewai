import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from benchmark.types import TaskSpec, TaskTier

class TaskRegistry:
    """Registry and loader for benchmark tasks."""

    def __init__(self, tasks_dir: Path):
        self.tasks_dir = tasks_dir
        self.tasks: Dict[str, TaskSpec] = {}

    def load_tasks(self):
        """Recursively load tasks from the tasks directory."""
        for yaml_path in self.tasks_dir.glob("**/task.yaml"):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                task_id = data["task"]["id"]
                
                # Mock loading some cases/tools for now
                # In a full implementation, we'd load these from the task directory
                self.tasks[task_id] = TaskSpec(
                    id=task_id,
                    name=data["task"]["name"],
                    version=data["task"]["version"],
                    tier=TaskTier(data["task"]["tier"]),
                    description=data["task"]["description"],
                    input_data={}, # To be loaded from inputs/
                    expected_output={}, # To be loaded from expected/
                    system_prompt=data["task"]["system_prompt"],
                    tools=data["task"].get("tools", [])
                )

    def get_task(self, task_id: str) -> Optional[TaskSpec]:
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[str]:
        return list(self.tasks.keys())
