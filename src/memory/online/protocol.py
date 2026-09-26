from __future__ import annotations

from typing import Any, Protocol

from .types import MemoryOutcome, MemoryTask


class OnlineMASMemory(Protocol):
    backend_name: str
    store: Any

    def set_api_manager(self, api_manager: Any) -> None: ...

    async def begin_task(self, task: MemoryTask) -> None: ...

    async def observe_trace(
        self,
        trace: dict[str, Any],
        *,
        agent_ids: tuple[str, ...],
    ) -> dict[str, list[str]]: ...

    async def retrieve_contexts(
        self,
        task: MemoryTask,
        *,
        top_k: int,
    ) -> dict[str, list[str]]: ...

    async def record_task(self, task: MemoryTask, outcome: MemoryOutcome) -> None: ...
