from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


@dataclass
class TaskInput:
    task: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MASState(TypedDict, total=False):
    task: TaskInput
    supervisor_brief: str
    subagent_outputs: Annotated[dict[str, str], merge_dicts]
    subagent_memory: Annotated[dict[str, list[str]], merge_dicts]
    autogen_action_history: list[str]
    final_output: str


# --------------- DyLAN-specific types ---------------


def _merge_layer_outputs(left: dict | None, right: dict | None) -> dict:
    """Merge per-layer outputs: {layer_idx: {agent_name: response}}."""
    result: dict = dict(left or {})
    for key, value in (right or {}).items():
        if key in result:
            result[key].update(value)
        else:
            result[key] = dict(value)
    return result


def _merge_active_mask(left: dict | None, right: dict | None) -> dict:
    """Merge active agent masks."""
    result = dict(left or {})
    result.update(right or {})
    return result


class DyLANState(TypedDict, total=False):
    task: TaskInput
    supervisor_brief: str
    layer_outputs: Annotated[dict[int, dict[str, str]], _merge_layer_outputs]
    active_mask: Annotated[dict[str, bool], _merge_active_mask]
    current_layer: int
    consensus_reached: bool
    final_output: str

