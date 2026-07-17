"""Append-only JSONL decision audit log."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .executors import ExecutionRecord


class DecisionLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, period: int, record: ExecutionRecord) -> None:
        row = {"period": period, **asdict(record)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
