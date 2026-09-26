from __future__ import annotations

import re
import string

from src.datasets.wikipedia import WikipediaActionExplorer, WikipediaServiceClient


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    return " ".join(text.split())


def compute_em(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def compute_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, gold_counts.get(token, 0))
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class HotpotQAEnv:
    def __init__(self, answer: str, max_steps: int = 10, client: WikipediaServiceClient | None = None) -> None:
        self.answer = answer
        self.max_steps = max_steps
        self.explorer = WikipediaActionExplorer(client=client)
        self.reward = 0.0
        self.final_answer: str | None = None
        self.steps_taken = 0
        self.final_f1 = 0.0

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
            return self.explorer.search(argument or ""), 0.0, False

        if action_type == "Lookup":
            return self.explorer.lookup(argument or ""), 0.0, False

        return "Invalid Action. Valid Actions are Search[<query>] Lookup[<title>] and Finish[<answer>].", -1.0, False

    def evaluate_finish(self, answer: str) -> tuple[str, float, bool]:
        self.steps_taken += 1
        final_answer = answer.strip()
        if normalize_answer(self.answer) in {"yes", "no"}:
            normalized_finish = normalize_answer(final_answer)
            first_token = normalized_finish.split(maxsplit=1)[0] if normalized_finish else ""
            if first_token in {"yes", "no"}:
                final_answer = first_token
        self.final_answer = final_answer
        exact_match = compute_em(self.final_answer, self.answer)
        self.final_f1 = compute_f1(self.final_answer, self.answer)
        self.reward = 1.0 if exact_match else 0.0
        if exact_match:
            return "Answer is CORRECT", self.reward, True
        return f"Answer is INCORRECT (gold: {self.answer})", self.reward, True

    def feedback(self) -> tuple[float, bool, str]:
        done = self.reward == 1.0
        message = "You successfully finished this task." if done else "You failed the task."
        return self.reward, done, message
