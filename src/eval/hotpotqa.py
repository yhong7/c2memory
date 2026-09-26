from dataclasses import dataclass, field
from typing import Any

from src.core.types import TaskInput
from src.eval.hotpotqa_env import compute_em, compute_f1


def evaluate_hotpotqa_output(task: TaskInput, result: dict[str, Any]) -> dict[str, Any]:
    gold = str(task.metadata.get("answer", "")).strip()
    predicted = str(result.get("predicted_answer") or result.get("final_output", "")).strip()
    em = compute_em(predicted, gold)
    f1 = compute_f1(predicted, gold)
    return {
        "dataset": "hotpotqa",
        "gold_answer": gold,
        "predicted_answer": predicted,
        "exact_match": em,
        "f1": f1,
        "reward": float(result.get("reward", 0.0) or 0.0),
        "done": result.get("done", False),
        "terminated": result.get("terminated", False),
        "steps": result.get("steps_taken", 0),
        "question_type": task.metadata.get("type", ""),
        "level": task.metadata.get("level", ""),
    }


@dataclass
class HotpotQADatasetEvaluator:
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        task: TaskInput,
        result: dict[str, Any],
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_ms: float = 0.0,
    ) -> dict[str, Any]:
        record = evaluate_hotpotqa_output(task, result)
        record.update(
            task_index=task.metadata.get("task_index"),
            task_id=task.metadata.get("id"),
            task_text=task.task,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
        )
        self.records.append(record)
        return record

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def em_count(self) -> int:
        return sum(1 for r in self.records if r.get("exact_match"))

    @property
    def avg_f1(self) -> float:
        if not self.records:
            return 0.0
        return sum(float(r.get("f1", 0.0) or 0.0) for r in self.records) / len(self.records)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(int(r.get("prompt_tokens", 0) or 0) for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(int(r.get("completion_tokens", 0) or 0) for r in self.records)

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {
                "tasks": 0,
                "exact_match": 0,
                "em_rate": 0.0,
                "avg_f1": 0.0,
                "avg_steps": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
            }

        task_count = len(self.records)
        by_type: dict[str, dict[str, float]] = {}
        for r in self.records:
            qtype = r.get("question_type", "unknown")
            if qtype not in by_type:
                by_type[qtype] = {"count": 0, "em": 0, "f1_sum": 0.0}
            by_type[qtype]["count"] += 1
            by_type[qtype]["em"] += 1 if r.get("exact_match") else 0
            by_type[qtype]["f1_sum"] += float(r.get("f1", 0.0) or 0.0)

        return {
            "tasks": task_count,
            "exact_match": self.em_count,
            "em_rate": self.em_count / task_count,
            "avg_f1": self.avg_f1,
            "avg_steps": sum(int(r.get("steps", 0) or 0) for r in self.records) / task_count,
            "by_type": {
                t: {
                    "count": int(v["count"]),
                    "em_rate": v["em"] / v["count"],
                    "avg_f1": v["f1_sum"] / v["count"],
                }
                for t, v in sorted(by_type.items())
            },
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
