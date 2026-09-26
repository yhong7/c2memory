import re
import string
from dataclasses import dataclass, field
from typing import Any

from src.core.types import TaskInput
from src.datasets.wikipedia import WikipediaActionExplorer, WikipediaServiceClient


class FeverEnv:
    def __init__(
        self,
        answer: str,
        max_steps: int = 7,
        client: WikipediaServiceClient | None = None,
    ) -> None:
        self.answer = answer
        self.max_steps = max_steps
        self.explorer = WikipediaActionExplorer(client=client)
        self.reward = 0.0
        self.final_answer: str | None = None
        self.steps_taken = 0

    @classmethod
    def process_action(cls, action: str) -> str:
        cleaned = action.strip().replace("<", "").replace(">", "")
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return ""
        first = lines[0]
        if first.lower().startswith("action:"):
            return first.split(":", 1)[1].strip()
        if ":" in first:
            return first.split(":", 1)[1].strip()
        return first

    @staticmethod
    def parse_action(action: str) -> tuple[str | None, str | None]:
        match = re.match(r"^(\w+)\[(.+)\]$", action)
        if not match:
            return None, None
        return match.group(1), match.group(2)

    def step(self, raw_action: str) -> tuple[str, float, bool]:
        action = self.process_action(raw_action)

        action_type, argument = self.parse_action(action)
        if action_type == "Finish":
            return self.evaluate_finish(argument or "")
        self.steps_taken += 1

        if action_type == "Search":
            observation = self.explorer.search(argument or "")
            return observation, 0.0, False

        if action_type == "Lookup":
            observation = self.explorer.lookup(argument or "")
            return observation, 0.0, False

        return "Invalid Action. Valid Actions are Search[<query>] Lookup[<title>] and Finish[<answer>].", -1.0, False

    def evaluate_finish(self, answer: str) -> tuple[str, float, bool]:
        self.steps_taken += 1
        self.final_answer = answer.strip()
        if match_exactly(self.final_answer, self.answer):
            self.reward = 1.0
            return "Answer is CORRECT", 1.0, True
        self.reward = 0.0
        return "Answer is INCORRECT", 0.0, True

    def feedback(self) -> tuple[float, bool, str]:
        done = self.reward == 1.0
        message = "You successfully finished this task." if done else "You failed the task."
        return self.reward, done, message


def normalize_answer(value: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(value.lower())))


def match_exactly(answer: str, key: str) -> bool:
    return normalize_answer(answer) == normalize_answer(key)


def evaluate_fever_output(task: TaskInput, result: dict[str, Any]) -> dict[str, Any]:
    gold = str(task.metadata.get("label", "")).strip()
    predicted = str(result.get("predicted_label") or "").strip()
    reward = float(result.get("reward", 0.0) or 0.0)
    return {
        "dataset": "fever",
        "gold_label": gold,
        "predicted_label": predicted,
        "correct": reward > 0.0,
        "reward": reward,
        "done": result.get("done", False),
        "terminated": result.get("terminated", False),
        "label_match": bool(predicted) and match_exactly(predicted, gold),
        "steps": result.get("steps_taken", 0),
    }


@dataclass
class FeverDatasetEvaluator:
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
        record = evaluate_fever_output(task, result)
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
    def correct_count(self) -> int:
        return sum(1 for record in self.records if record.get("correct"))

    @property
    def reward_sum(self) -> float:
        return sum(float(record.get("reward", 0.0) or 0.0) for record in self.records)

    @property
    def done_count(self) -> int:
        return sum(1 for record in self.records if record.get("done"))

    @property
    def terminated_count(self) -> int:
        return sum(1 for record in self.records if record.get("terminated"))

    @property
    def total_prompt_tokens(self) -> int:
        return sum(int(record.get("prompt_tokens", 0) or 0) for record in self.records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(int(record.get("completion_tokens", 0) or 0) for record in self.records)

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {
                "tasks": 0,
                "correct": 0,
                "accuracy": 0.0,
                "avg_reward": 0.0,
                "done_rate": 0.0,
                "terminated_rate": 0.0,
                "avg_steps": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
            }

        task_count = len(self.records)
        return {
            "tasks": task_count,
            "correct": self.correct_count,
            "accuracy": self.correct_count / task_count,
            "avg_reward": self.reward_sum / task_count,
            "done_rate": self.done_count / task_count,
            "terminated_rate": self.terminated_count / task_count,
            "avg_steps": sum(int(record.get("steps", 0) or 0) for record in self.records) / task_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
