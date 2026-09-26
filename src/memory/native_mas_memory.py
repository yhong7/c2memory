from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.types import TaskInput

from .online.factory import (
    BACKEND_ALIASES,
    BACKEND_TYPES,
    create_online_memory,
    normalize_backend_name,
)
from .online.protocol import OnlineMASMemory
from .online.types import MemoryOutcome, MemoryTask


NATIVE_MAS_MEMORY_BACKENDS = frozenset(BACKEND_TYPES)
NATIVE_MAS_MEMORY_CHOICES = (
    "none",
    *BACKEND_TYPES,
    *BACKEND_ALIASES,
)
NATIVE_MAS_MEMORY_VERSION = "c2mem-paper-v2"


def normalize_mas_memory_backend(backend: str) -> str:
    return normalize_backend_name(backend)


def _memory_task(task: TaskInput, agent_ids: list[str]) -> MemoryTask:
    return MemoryTask(
        text=task.task,
        metadata=dict(task.metadata),
        agent_ids=tuple(agent_ids),
    )


@dataclass
class NativeMASMemoryStore:
    """Runtime-facing adapter for independently implemented online memories."""

    backend: str
    persist_dir: Path
    implementation: OnlineMASMemory

    @classmethod
    def load(cls, *, backend: str, persist_dir: str | Path) -> "NativeMASMemoryStore":
        canonical = normalize_mas_memory_backend(backend)
        root = Path(persist_dir)
        implementation = create_online_memory(
            backend=canonical,
            root=root,
            api_manager=None,
        )
        return cls(backend=canonical, persist_dir=root, implementation=implementation)

    @property
    def records(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.implementation.store.records]

    @property
    def state(self) -> dict[str, Any]:
        return self.implementation.store.state

    async def begin_task(
        self,
        task: TaskInput,
        *,
        agent_ids: list[str],
        api_manager: Any,
    ) -> None:
        self.implementation.set_api_manager(api_manager)
        await self.implementation.begin_task(_memory_task(task, agent_ids))

    async def observe_trace(
        self,
        trace: dict[str, Any],
        *,
        agent_ids: list[str],
        api_manager: Any,
    ) -> dict[str, list[str]]:
        self.implementation.set_api_manager(api_manager)
        return await self.implementation.observe_trace(
            dict(trace),
            agent_ids=tuple(agent_ids),
        )

    async def retrieve_contexts(
        self,
        task: TaskInput,
        *,
        agent_ids: list[str],
        api_manager: Any,
        top_k: int = 2,
    ) -> dict[str, list[str]]:
        self.implementation.set_api_manager(api_manager)
        return await self.implementation.retrieve_contexts(
            _memory_task(task, agent_ids),
            top_k=top_k,
        )

    async def record_task(
        self,
        task: TaskInput,
        result: dict[str, Any],
        *,
        agent_traces: list[dict[str, Any]],
        success: bool,
        api_manager: Any,
    ) -> None:
        self.implementation.set_api_manager(api_manager)
        feedback = str(
            result.get("feedback")
            or result.get("final_feedback")
            or result.get("observation")
            or ""
        )
        await self.implementation.record_task(
            _memory_task(task, []),
            MemoryOutcome(
                result=dict(result),
                agent_traces=tuple(dict(trace) for trace in agent_traces),
                success=bool(success),
                feedback=feedback,
            ),
        )
