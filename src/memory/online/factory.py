from __future__ import annotations

from pathlib import Path
from typing import Any

from .c2mem import C2Memory
from .protocol import OnlineMASMemory


BACKEND_ALIASES: dict[str, str] = {}

BACKEND_TYPES = {
    "c2mem": C2Memory,
}


def normalize_backend_name(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return BACKEND_ALIASES.get(normalized, normalized)


def create_online_memory(
    *,
    backend: str,
    root: str | Path,
    api_manager: Any,
) -> OnlineMASMemory:
    normalized = normalize_backend_name(backend)
    backend_type = BACKEND_TYPES.get(normalized)
    if backend_type is None:
        raise ValueError(f"unsupported online MAS memory backend: {backend}")
    return backend_type(root=root, api_manager=api_manager)
