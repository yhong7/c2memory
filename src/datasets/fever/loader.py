import json
from pathlib import Path

from src.core.types import TaskInput


DEFAULT_FEVER_PATH = Path("data/fever/fever_dev.jsonl")


def load_fever_tasks(
    *,
    limit: int = 100,
    path: str | Path = DEFAULT_FEVER_PATH,
) -> list[TaskInput]:
    dataset_path = Path(path)
    tasks: list[TaskInput] = []

    with dataset_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            row = json.loads(line)
            tasks.append(
                TaskInput(
                    task=row["claim"],
                    metadata={
                        "dataset": "fever",
                        "task_index": idx,
                        "id": row.get("id"),
                        "label": row.get("label"),
                        "verifiable": row.get("verifiable"),
                        "env_name": "fever",
                    },
                )
            )

    return tasks
