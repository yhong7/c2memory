from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryTask:
    text: str
    metadata: dict[str, Any]
    agent_ids: tuple[str, ...]

    @property
    def query(self) -> str:
        for key in ("question", "problem", "task", "claim", "task_goal", "goal"):
            value = str(self.metadata.get(key, "")).strip()
            if value:
                return value
        return self.text[:12000]


@dataclass(frozen=True)
class MemoryOutcome:
    result: dict[str, Any]
    agent_traces: tuple[dict[str, Any], ...]
    success: bool
    feedback: str = ""


@dataclass
class MemoryRecord:
    memory_id: str
    task: str
    label: bool
    trajectory: str
    feedback: str
    embedding_text: str
    embedding: list[float] = field(default_factory=list)
    task_summary: str = ""
    key_steps: str = ""
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "native_mas_memory",
            "memory_id": self.memory_id,
            "task": self.task,
            "label": self.label,
            "trajectory": self.trajectory,
            "feedback": self.feedback,
            "embedding_text": self.embedding_text,
            "embedding": self.embedding,
            "task_summary": self.task_summary,
            "key_steps": self.key_steps,
            "failure_reason": self.failure_reason,
            "metadata": self.metadata,
            "created_index": self.created_index,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=str(value.get("memory_id", "")),
            task=str(value.get("task", "")),
            label=bool(value.get("label", False)),
            trajectory=str(value.get("trajectory", "")),
            feedback=str(value.get("feedback", "")),
            embedding_text=str(value.get("embedding_text", "")),
            embedding=[float(item) for item in value.get("embedding", []) or []],
            task_summary=str(value.get("task_summary", "")),
            key_steps=str(value.get("key_steps", "")),
            failure_reason=str(value.get("failure_reason", "")),
            metadata=dict(value.get("metadata", {}) or {}),
            created_index=int(value.get("created_index", 0) or 0),
        )
