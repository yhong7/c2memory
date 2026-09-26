from __future__ import annotations

from collections.abc import Callable


ObservationFormatter = Callable[[str, str], str]


def _default_observation_formatter(observation: str, action: str) -> str:
    return str(observation or "").strip()


def render_interactive_step_prompt(
    *,
    system_prompt: str,
    goal_label: str,
    goal: str,
    trajectory: list[dict[str, str]],
    executable_actions: list[str] | None = None,
    current_observation: str | None = None,
    observation_formatter: ObservationFormatter | None = None,
    omit_latest_observation_from_history: bool = False,
    final_instruction: str = "Return exactly two lines: Action and Reason.",
) -> str:
    """Render the standard task-layer state for one interactive MAS step.

    The task layer owns environment-specific state: the task goal, current
    observation, executable actions, and committed action/observation history.
    MAS and memory layers should consume this rendered state without knowing
    which environment produced it.
    """
    formatter = observation_formatter or _default_observation_formatter
    lines: list[str] = [f"{goal_label}: {goal}"]

    if current_observation is not None:
        rendered_current = str(current_observation or "").strip()
        lines.extend(["", "Current observation:"])
        lines.append(rendered_current if rendered_current else "(empty)")

    lines.extend(["", "Executable actions:"])
    if executable_actions:
        for action in executable_actions:
            rendered_action = str(action or "").strip()
            if rendered_action:
                lines.append(f"- {rendered_action}")
    else:
        lines.append("(not provided)")

    lines.extend(["", "Committed trajectory:"])
    if not trajectory:
        lines.append("(none yet)")
    else:
        last_index = len(trajectory)
        for idx, turn in enumerate(trajectory, start=1):
            action = str(turn.get("action", ""))
            lines.append(f"Action {idx}: {action}")
            if omit_latest_observation_from_history and idx == last_index:
                continue
            observation = formatter(str(turn.get("observation", "")), action)
            if observation:
                lines.append(f"Observation {idx}: {observation}")

    return (
        f"{system_prompt.strip()}\n\n"
        f"{chr(10).join(lines)}\n\n"
        f"{final_instruction}"
    )
