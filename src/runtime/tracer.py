"""Generic trace-/run-recording for any LangGraph graph topology.

Records per‑round I/O, token usage, and timing, then computes aggregate
statistics across rounds (success rate, token totals, etc.).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class RoundRecord:
    round_idx: int
    task_input: str
    final_output: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float
    success: bool = False
    predicted_answer: str = ""
    done: bool = False
    terminated: bool = False
    feedback: str = ""
    steps_taken: int | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenEvent:
    name: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecorder:
    graph_name: str
    model_name: str
    max_rounds: int = 1

    rounds: list[RoundRecord] = field(default_factory=list)
    token_events: list[TokenEvent] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── per‑round helpers ──────────────────────────────────────────
    def record_round(self, record: RoundRecord) -> None:
        self.rounds.append(record)

    def record_token_event(self, event: TokenEvent) -> None:
        if event.prompt_tokens or event.completion_tokens:
            self.token_events.append(event)

    # ── aggregate statistics ───────────────────────────────────────
    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.rounds) + sum(e.prompt_tokens for e in self.token_events)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.rounds) + sum(e.completion_tokens for e in self.token_events)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.rounds if r.success)

    @property
    def success_rate(self) -> float:
        if not self.rounds:
            return 0.0
        return self.success_count / len(self.rounds)

    @property
    def avg_duration_ms(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(r.duration_ms for r in self.rounds) / len(self.rounds)

    @property
    def total_duration_ms(self) -> float:
        return sum(r.duration_ms for r in self.rounds)

    @property
    def avg_output_length(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(len(r.final_output) for r in self.rounds) / len(self.rounds)

    # ── summary dict / pretty-print ────────────────────────────────
    def summary(self) -> dict[str, Any]:
        return {
            "graph": self.graph_name,
            "model": self.model_name,
            "max_rounds": self.max_rounds,
            "completed_rounds": len(self.rounds),
            "started_at": self.started_at,
            "success_count": self.success_count,
            "success_rate": round(self.success_rate, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_duration_ms": round(self.avg_duration_ms, 1),
            "total_duration_ms": round(self.total_duration_ms, 1),
            "avg_output_length": round(self.avg_output_length, 1),
        }

    def format_table(self) -> str:
        lines: list[str] = []
        lines.append(f"{'#':>3} {'input':>40} {'output':>8} {'prompt':>8} {'compl':>8} {'dur_ms':>8} {'ok':>4}")
        lines.append("-" * 90)
        for r in self.rounds:
            inp_preview = r.task_input[:37] + "..." if len(r.task_input) > 40 else r.task_input
            lines.append(
                f"{r.round_idx:>3} {inp_preview:>40} {len(r.final_output):>8} "
                f"{r.prompt_tokens:>8} {r.completion_tokens:>8} {r.duration_ms:>8.0f} {str(r.success):>4}"
            )
        for e in self.token_events:
            lines.append(
                f"{'ev':>3} {e.name[:40]:>40} {'':>8} "
                f"{e.prompt_tokens:>8} {e.completion_tokens:>8} {e.duration_ms:>8.0f} {'':>4}"
            )
        lines.append("-" * 90)
        s = self.summary()
        lines.append(
            f"Σ rounds={s['completed_rounds']}/{s['max_rounds']}  "
            f"success={s['success_count']} ({s['success_rate']:.1%})  "
            f"tokens={s['total_tokens']} (prompt={s['total_prompt_tokens']} compl={s['total_completion_tokens']})  "
            f"dur={s['total_duration_ms']:.0f}ms (avg {s['avg_duration_ms']:.0f}ms/round)"
        )
        return "\n".join(lines)
