import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from benchmark.types import RunMeasurement, Framework

class ResultsDatabase:
    """SQLite-backed storage for high-volume benchmark results."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    framework TEXT,
                    iteration INTEGER,
                    timestamp TEXT,
                    success BOOLEAN,
                    score FLOAT,
                    latency_ms FLOAT,
                    tokens INTEGER,
                    cost_usd FLOAT,
                    raw_data JSON
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_framework ON runs(task_id, framework)")

    def save_run(self, result: RunMeasurement):
        """Insert or replace a run measurement."""
        # Custom serializer for the JSON blob
        def serializer(obj):
            if isinstance(obj, datetime): return obj.isoformat()
            if hasattr(obj, "__dict__"): return obj.__dict__
            return str(obj)

        raw_data = json.dumps(result.__dict__, default=serializer)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs 
                (run_id, task_id, framework, iteration, timestamp, success, score, latency_ms, tokens, cost_usd, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.run_id,
                result.task_id,
                result.framework.value,
                result.iteration,
                result.timestamp.isoformat(),
                result.task_success,
                result.task_score,
                result.total_latency_ms,
                result.total_tokens,
                result.total_cost_usd,
                raw_data
            ))

    def get_task_results(self, task_id: str, framework: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve results for analysis."""
        query = "SELECT raw_data FROM runs WHERE task_id = ?"
        params = [task_id]
        if framework:
            query += " AND framework = ?"
            params.append(framework)
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return [json.loads(row[0]) for row in cursor.fetchall()]

    def list_tasks(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT task_id FROM runs")
            return [row[0] for row in cursor.fetchall()]
