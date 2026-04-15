import docker
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from benchmark.types import ResourceSample

class DockerManager:
    """Manages Docker container lifecycle for isolated benchmark runs."""

    def __init__(self, config: Dict[str, Any]):
        try:
            self.client = docker.from_env()
            self.docker_available = True
        except Exception:
            self.client = None
            self.docker_available = False
            print("Warning: Docker not available. Containerized runs will fail.")
        
        self.config = config.get("docker", {})
        self.base_image = self.config.get("base_image")
        self.network = self.config.get("network")

    def run_container(
        self, 
        image: str, 
        command: List[str], 
        env: Dict[str, str],
        volumes: Dict[str, Dict[str, str]] = {}
    ) -> Dict[str, Any]:
        """Run a command in a fresh container and monitor resources."""
        container = self.client.containers.run(
            image,
            command=command,
            environment=env,
            network=self.config["network"],
            mem_limit=self.config["memory_limit"],
            cpu_quota=int(float(self.config["cpu_limit"]) * 100000),
            volumes=volumes,
            detach=True,
            remove=False
        )

        resource_history = []
        peak_memory = 0.0
        
        try:
            # Resource monitoring loop
            while True:
                container.reload()
                if container.status != "running":
                    break
                
                try:
                    stats = container.stats(stream=False)
                    # Parse Stats
                    mem_usage = stats["memory_stats"].get("usage", 0) / (1024 * 1024) # MB
                    
                    # CPU aggregation depends on system architecture
                    cpu_percent = self._calculate_cpu_percent(stats)

                    peak_memory = max(peak_memory, mem_usage)
                    resource_history.append(ResourceSample(
                        timestamp=datetime.utcnow(),
                        cpu_percent=cpu_percent,
                        memory_mb=mem_usage
                    ))
                except (KeyError, docker.errors.APIError):
                    # Container might be stopping
                    break
                    
                time.sleep(0.1)

            result = container.wait()
            logs = container.logs().decode("utf-8")
            
            return {
                "exit_code": result["StatusCode"],
                "logs": logs,
                "peak_memory_mb": peak_memory,
                "resource_history": resource_history
            }
        finally:
            container.remove()

    def _calculate_cpu_percent(self, stats: Dict[str, Any]) -> float:
        """Helper to calculate CPU percentage from Docker stats."""
        try:
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
            num_cpus = stats["cpu_stats"].get("online_cpus", len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1])))
            
            if system_delta > 0 and cpu_delta > 0:
                return (cpu_delta / system_delta) * num_cpus * 100.0
            return 0.0
        except KeyError:
            return 0.0

    def build_image(self, dockerfile_path: Path, tag: str):
        """Build a framework-specific image."""
        print(f"Building image {tag} from {dockerfile_path}...")
        self.client.images.build(path=str(dockerfile_path.parent), dockerfile=dockerfile_path.name, tag=tag)
