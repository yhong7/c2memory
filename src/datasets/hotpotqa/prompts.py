import json
import re

from src.datasets.interactive import render_interactive_step_prompt


hotpotqa_system_prompt = """You are a multi-hop retrieval agent for HotpotQA.
The overall system solves the question by committing one next environment action at a time.

Environment actions:
1. Search[entity or query]
2. Lookup[title]
3. Finish[answer]

Rules:
- Use Search to retrieve the top candidate Wikipedia titles for one hop of the question.
- Use Lookup to fetch the full metadata for one exact title returned by Search or discovered in page links.
- Only use Finish when you have enough evidence to answer.
- Finish only the outermost answer slot, never an intermediate clue entity; use the shortest exact span.
- For yes/no, verify both entities and use exactly Finish[yes] or Finish[no].
- Return exactly two lines and nothing else:
Action: Search[query] or Lookup[title] or Finish[answer]
Reason: one short sentence explaining why this is the next step.
- Do not output free-form explanations or multiple actions."""


hotpotqa_few_shots: list[str] = []

HOTPOTQA_EXECUTABLE_ACTIONS = [
    "Search[entity or query]",
    "Lookup[title]",
    "Finish[answer]",
]


def _keywords(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) >= 4
    }


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _compact_document(title: str, document: str, keywords: set[str], max_chars: int) -> str:
    sentences = _split_sentences(document)
    selected: list[str] = []
    for sentence in sentences[:2]:
        if sentence not in selected:
            selected.append(sentence)

    scored: list[tuple[int, int, str]] = []
    for idx, sentence in enumerate(sentences[2:], start=2):
        sentence_tokens = _keywords(sentence)
        score = len(sentence_tokens & keywords)
        if score:
            scored.append((score, -idx, sentence))
    for _, _, sentence in sorted(scored, reverse=True):
        if sentence not in selected:
            selected.append(sentence)
        if len(" ".join(selected)) >= max_chars:
            break

    compact = f"title: {title}\n" + " ".join(selected)
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "\n...[observation truncated]"


def _compact_observation(observation: str, *, question: str = "", action: str = "", max_chars: int = 1200) -> str:
    text = str(observation or "").strip()
    keywords = _keywords(f"{question} {action}")
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            title = str(payload.get("title", "")).strip()
            document = str(payload.get("document", "")).strip()
            if document:
                return _compact_document(title, document, keywords, max_chars)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[observation truncated]"


def build_hotpotqa_step_prompt(question: str, trajectory: list[dict[str, str]]) -> str:
    """Build a prompt for one interactive HotpotQA retrieval step."""
    return render_interactive_step_prompt(
        system_prompt=hotpotqa_system_prompt,
        goal_label="Question",
        goal=question,
        trajectory=trajectory,
        executable_actions=HOTPOTQA_EXECUTABLE_ACTIONS,
        observation_formatter=lambda observation, action: _compact_observation(
            observation,
            question=question,
            action=action,
        ),
    )
