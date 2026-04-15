from pathlib import Path
from typing import Dict, List, Any, Optional
from benchmark.analysis import calculate_summary, compare_frameworks
from benchmark.types import Framework
from benchmark.results_db import ResultsDatabase

class MarkdownReporter:
    """Generates human-readable Markdown reports from benchmark results."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.db = ResultsDatabase(results_dir / "benchmarks.sqlite")

    def generate(self, output_path: Path):
        """Aggregate results from SQLite and generate a full MD report."""
        task_ids = self.db.list_tasks()
        
        grouped_data = {}
        for task_id in task_ids:
            results = self.db.get_task_results(task_id)
            for r in results:
                f_name = r["framework"]
                if task_id not in grouped_data:
                    grouped_data[task_id] = {}
                if f_name not in grouped_data[task_id]:
                    grouped_data[task_id][f_name] = {
                        "latency": [],
                        "tokens": [],
                        "cost": [],
                        "score": []
                    }
                
                grouped_data[task_id][f_name]["latency"].append(r["total_latency_ms"])
                grouped_data[task_id][f_name]["tokens"].append(r["total_tokens"])
                grouped_data[task_id][f_name]["cost"].append(r["total_cost_usd"])
                grouped_data[task_id][f_name]["score"].append(r["task_score"])

        report_lines = [
            "# Agent Framework Benchmark Report",
            f"Generated: {Path(output_path).name} at {Path(output_path).stat().st_mtime if Path(output_path).exists() else ''}",
            "## Summary of Results",
            ""
        ]

        for task_id, benchmarks in grouped_data.items():
            report_lines.append(f"### Task: {task_id}")
            report_lines.append("| Framework | Iterations | Avg Latency (ms) | Avg Tokens | Avg Cost ($) | Success Rate |")
            report_lines.append("|-----------|------------|------------------|------------|--------------|--------------|")
            
            f_names = list(benchmarks.keys())
            for framework in f_names:
                metrics = benchmarks[framework]
                n = len(metrics["latency"])
                avg_lat = sum(metrics["latency"]) / n if n > 0 else 0
                avg_tok = sum(metrics["tokens"]) / n if n > 0 else 0
                avg_cost = sum(metrics["cost"]) / n if n > 0 else 0
                success_rate = (sum(1 for s in metrics["score"] if s >= 0.7) / n) * 100 if n > 0 else 0
                
                report_lines.append(f"| {framework} | {n} | {avg_lat:.2f} | {avg_tok:.0f} | {avg_cost:.4f} | {success_rate:.1f}% |")
            
            report_lines.append("\n#### Statistical Significance (Latency)")
            if len(f_names) >= 2:
                report_lines.append("| Comparison | P-Value | Significant? | Effect Size | Winner |")
                report_lines.append("|------------|---------|--------------|-------------|--------|")
                for i in range(len(f_names)):
                    for j in range(i + 1, len(f_names)):
                        fa, fb = f_names[i], f_names[j]
                        comp = compare_frameworks(
                            benchmarks[fa]["latency"], 
                            benchmarks[fb]["latency"], 
                            fa, fb, task_id, "latency"
                        )
                        sig_str = "**Yes**" if comp.is_significant else "No"
                        winner_str = comp.winner if comp.is_significant else "---"
                        report_lines.append(f"| {fa} vs {fb} | {comp.p_value:.4f} | {sig_str} | {comp.effect_magnitude} ({comp.effect_size:.2f}) | {winner_str} |")
            else:
                report_lines.append("Insufficient frameworks for comparison.")
            
            report_lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(report_lines))
