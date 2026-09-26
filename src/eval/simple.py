from typing import Any

from src.core.types import TaskInput
from src.eval.fever import evaluate_fever_output


def evaluate_final_output(output: str) -> dict[str, Any]:
    return {
        "non_empty": bool(output.strip()),
        "length": len(output.strip()),
    }


def evaluate_task_result(task: TaskInput, result: dict[str, Any]) -> dict[str, Any]:
    if task.metadata.get("dataset") == "fever":
        return evaluate_fever_output(task, result)
    if task.metadata.get("dataset") == "hotpotqa":
        from src.eval.hotpotqa import evaluate_hotpotqa_output
        return evaluate_hotpotqa_output(task, result)
    return evaluate_final_output(result.get("final_output", ""))
