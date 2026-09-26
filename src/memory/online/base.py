from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .store import JSONMemoryStore
from .types import MemoryOutcome, MemoryTask


def render_trajectory(outcome: MemoryOutcome) -> str:
    lines: list[str] = []
    for trace in outcome.agent_traces:
        agent = str(trace.get("agent", "agent")).strip()
        output = str(trace.get("output", "")).strip()
        if output:
            lines.append(f"[{agent}] {output[:1800]}")
    for turn in outcome.result.get("trajectory", []) or []:
        if not isinstance(turn, dict):
            continue
        action = str(turn.get("action", "")).strip()
        observation = str(turn.get("observation", "")).strip()
        if action or observation:
            lines.append(f"[environment] action={action[:500]} observation={observation[:1200]}")
    if not lines:
        final_output = str(outcome.result.get("final_output", "")).strip()
        if final_output:
            lines.append(f"[final] {final_output[:1800]}")
    return "\n".join(lines)


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return -1.0
    return dot / (left_norm * right_norm)


class BaseOnlineMemory:
    backend_name = "base"

    def __init__(self, *, root: str | Path, api_manager: Any) -> None:
        self.api_manager = api_manager
        self.store = JSONMemoryStore(root, self.backend_name)

    def set_api_manager(self, api_manager: Any) -> None:
        self.api_manager = api_manager

    async def begin_task(self, task: MemoryTask) -> None:
        return None

    async def observe_trace(
        self,
        trace: dict[str, Any],
        *,
        agent_ids: tuple[str, ...],
    ) -> dict[str, list[str]]:
        return {}

    async def _embed(self, text: str) -> list[float]:
        vectors = await self.api_manager.embed_texts([str(text)[:12000]])
        if len(vectors) != 1:
            raise RuntimeError(f"{self.backend_name} embedding returned {len(vectors)} vectors")
        return [float(item) for item in vectors[0]]

    async def _chat(self, system: str, user: str, *, max_tokens: int = 500) -> str:
        return str(
            await self.api_manager.chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user[:24000]},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
        ).strip()
