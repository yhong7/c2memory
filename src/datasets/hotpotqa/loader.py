import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.core.types import TaskInput


DEFAULT_HOTPOTQA_PATH = Path("data/hotpotqa/hotpot_dev_distractor_v1.json")


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"HotpotQA file must contain a JSON array: {path}")
    for row in payload:
        if isinstance(row, dict):
            yield row


def _supporting_titles(value: Any) -> list[str]:
    if isinstance(value, dict):
        candidates = value.get("title", []) or []
    elif isinstance(value, list):
        candidates = [
            item[0] if isinstance(item, (list, tuple)) and item else item.get("title")
            for item in value
            if isinstance(item, (list, tuple, dict))
        ]
    else:
        candidates = []

    titles: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        title = str(value or "").strip()
        key = title.casefold()
        if title and key not in seen:
            seen.add(key)
            titles.append(title)
    return titles


def load_hotpotqa_tasks(
    *,
    limit: int = 100,
    path: str | Path = DEFAULT_HOTPOTQA_PATH,
) -> list[TaskInput]:
    """Load a locally supplied HotpotQA distractor split.

    The loader expects an explicit local path and does not download the dataset
    automatically.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            "HotpotQA data is not bundled and automatic remote download is disabled. "
            f"Obtain the standard distractor split independently and place it at {dataset_path}."
        )

    tasks: list[TaskInput] = []
    for idx, row in enumerate(_iter_rows(dataset_path)):
        if limit is not None and idx >= limit:
            break
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        tasks.append(
            TaskInput(
                task=question,
                metadata={
                    "dataset": "hotpotqa",
                    "task_index": idx,
                    "id": row.get("id") or row.get("_id"),
                    "answer": answer,
                    "type": row.get("type"),
                    "level": row.get("level"),
                    "question": question,
                    "supporting_titles": _supporting_titles(row.get("supporting_facts")),
                },
            )
        )
    return tasks
