"""Checkpoint management system for NBA data ingestion.

Tracks completed and failed download tasks to provide fault tolerance,
automatic resume, and idempotent execution.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

MANIFEST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "ingestion_manifest.json"
)


class CheckpointManager:
    def __init__(self, manifest_path: str = MANIFEST_PATH):
        self.manifest_path = os.path.abspath(manifest_path)
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        self.manifest: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[CheckpointManager] Warning: Failed to load manifest ({e}). Creating new.")
                return {}
        return {}

    def _save(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def is_completed(self, task_key: str) -> bool:
        task = self.manifest.get(task_key)
        return task is not None and task.get("status") == "COMPLETED"

    def record_success(self, task_key: str, file_path: str, row_count: int, metadata: Optional[Dict] = None) -> None:
        self.manifest[task_key] = {
            "status": "COMPLETED",
            "file_path": os.path.abspath(file_path),
            "row_count": row_count,
            "completed_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._save()

    def record_failure(self, task_key: str, error_message: str) -> None:
        self.manifest[task_key] = {
            "status": "FAILED",
            "error": error_message,
            "failed_at": datetime.now().isoformat(),
        }
        self._save()

    def get_summary(self) -> Dict[str, Any]:
        total = len(self.manifest)
        completed = sum(1 for v in self.manifest.values() if v.get("status") == "COMPLETED")
        failed = sum(1 for v in self.manifest.values() if v.get("status") == "FAILED")
        total_rows = sum(v.get("row_count", 0) for v in self.manifest.values() if v.get("status") == "COMPLETED")
        return {
            "total_tasks": total,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "total_records_ingested": total_rows,
        }
