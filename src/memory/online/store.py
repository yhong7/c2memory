from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import MemoryRecord


class JSONMemoryStore:
    def __init__(self, root: str | Path, backend_name: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend_name = backend_name
        self.records_path = self.root / "records.jsonl"
        self.state_path = self.root / "state.json"
        self.records = self._load_records()
        self.state = self._load_state()
        self._validate_backend()

    def _validate_backend(self) -> None:
        stored_backend = str(self.state.get("backend", "")).strip()
        if stored_backend and stored_backend != self.backend_name:
            raise ValueError(
                "memory store backend mismatch: "
                f"requested {self.backend_name!r}, found {stored_backend!r} in {self.state_path}"
            )
        for record in self.records:
            record_backend = str(record.metadata.get("backend", "")).strip()
            if record_backend and record_backend != self.backend_name:
                raise ValueError(
                    "memory record backend mismatch: "
                    f"requested {self.backend_name!r}, found {record_backend!r} "
                    f"in record {record.memory_id!r}"
                )
        self.state["backend"] = self.backend_name

    def _load_records(self) -> list[MemoryRecord]:
        if not self.records_path.exists():
            return []
        records: list[MemoryRecord] = []
        with self.records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(MemoryRecord.from_dict(json.loads(line)))
        return records

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"backend": self.backend_name, "insights": []}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def append(self, record: MemoryRecord) -> None:
        self.records.append(record)
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self.state["record_count"] = len(self.records)
        self.save_state()

    def save_records(self) -> None:
        temp_path = self.records_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        temp_path.replace(self.records_path)

    def save_state(self) -> None:
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.state_path)
