from src.core.types import TaskInput
from src.datasets.fever import load_fever_tasks
from src.datasets.hotpotqa import load_hotpotqa_tasks


def load_dataset_tasks(dataset: str, limit: int | None = None) -> list[TaskInput]:
    if dataset == "fever":
        return load_fever_tasks(limit=limit) if limit is not None else load_fever_tasks()
    if dataset == "hotpotqa":
        return load_hotpotqa_tasks(limit=limit) if limit is not None else load_hotpotqa_tasks()
    raise ValueError(
        f"Unsupported dataset: {dataset}. Available datasets: fever, hotpotqa"
    )
