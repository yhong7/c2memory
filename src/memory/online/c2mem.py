from __future__ import annotations

import json
import math
import os
import random
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from .base import BaseOnlineMemory, cosine, render_trajectory
from .types import MemoryOutcome, MemoryRecord, MemoryTask


EPISODE_SUMMARY_SYSTEM = """You are the episode summarizer for C2Mem.
Summarize one completed multi-agent execution as reusable memory.
Separate task content from execution behavior. The behavior pattern is the primary
routing representation and must describe the ordered control mechanism rather than
the task topic or answer.
Return only valid JSON with exactly these keys:
{
  "content_summary": "short task and decisive-evidence summary",
  "behavior_pattern": "Trigger: ... Early: ... Feedback: ... Transition: ... Stop: ...",
  "outcome_summary": "why this execution succeeded or failed",
  "reusable_experience": {"agent_id": "applicability trigger plus next-action rule"}
}
Use actual agent IDs when they are visible. Keep all values entity-free except
content_summary. Do not include markdown."""


EVALUATOR_SYSTEM = """You are the C2Mem behavioral-support evaluator.
Given a current execution prefix and retrieved behavior-tree nodes, assign a joint
probability distribution over the listed ordinary node IDs and node 0.
An ordinary weight is the behavioral support that node provides for the current
execution. Weight on 0 means the retrieved memories provide insufficient support.
Judge ordered execution compatibility, not topic or entity overlap.
Return only JSON:
{"weights": {"node_id": 0.0, "0": 0.0}, "rationale": "short reason"}
Include every listed ID exactly once. All weights must be nonnegative and sum to 1."""


NODE_SUMMARY_SYSTEM = """You maintain one C2Mem behavior-tree node.
Summarize the recurring ordered execution mechanism shared by its member episodes.
Do not summarize task topics, entities, answers, or correctness.
Return only JSON:
{
  "behavior_pattern": "Trigger: ... Early: ... Feedback: ... Transition: ... Stop: ...",
  "reusable_experience": {"agent_id": "applicability trigger plus next-action rule"}
}"""


UNKNOWN_NODE_ID = "0"
C2MEM_STATE_SCHEMA = "c2mem-paper-v2"


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))


def _json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _agent_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(agent_id).strip(): str(rule).strip()
        for agent_id, rule in value.items()
        if str(agent_id).strip() and str(rule).strip()
    }


def _normalize_weights(
    raw_weights: Any,
    candidate_ids: list[str],
    *,
    fallback_similarities: dict[str, float],
) -> dict[str, float]:
    allowed = [*candidate_ids, UNKNOWN_NODE_ID]
    parsed = raw_weights if isinstance(raw_weights, dict) else {}
    weights: dict[str, float] = {}
    for node_id in allowed:
        try:
            weights[node_id] = max(0.0, float(parsed.get(node_id, 0.0) or 0.0))
        except (TypeError, ValueError):
            weights[node_id] = 0.0
    total = sum(weights.values())
    if total <= 0:
        best_similarity = max(fallback_similarities.values(), default=-1.0)
        ordinary = {
            node_id: max(0.0, fallback_similarities.get(node_id, -1.0) + 1.0)
            for node_id in candidate_ids
        }
        unknown = max(0.1, 1.0 - max(0.0, best_similarity))
        total = sum(ordinary.values()) + unknown
        weights = {
            node_id: value / total
            for node_id, value in {**ordinary, UNKNOWN_NODE_ID: unknown}.items()
        }
        return weights
    return {node_id: value / total for node_id, value in weights.items()}


def _node_path(node_id: str, nodes_by_id: dict[str, dict[str, Any]]) -> list[str]:
    path: list[str] = []
    current = str(node_id)
    seen: set[str] = set()
    while current and current not in seen and current in nodes_by_id:
        seen.add(current)
        path.append(current)
        current = str(nodes_by_id[current].get("parent_id", "")).strip()
    path.append("__virtual_root__")
    return list(reversed(path))


def normalized_tree_distance(
    left_id: str,
    right_id: str,
    nodes: list[dict[str, Any]],
) -> float:
    if left_id == right_id:
        return 0.0
    nodes_by_id = {
        str(node.get("node_id", "")).strip(): node
        for node in nodes
        if str(node.get("node_id", "")).strip()
    }
    left_path = _node_path(left_id, nodes_by_id)
    right_path = _node_path(right_id, nodes_by_id)
    shared = 0
    for left, right in zip(left_path, right_path):
        if left != right:
            break
        shared += 1
    edge_distance = (len(left_path) - shared) + (len(right_path) - shared)
    maximum = max(1, (len(left_path) - 1) + (len(right_path) - 1))
    return _clip(edge_distance / maximum)


def compute_c2_metrics(
    *,
    weights: dict[str, float],
    nodes: list[dict[str, Any]],
    unknown: dict[str, Any],
    commitment_lambda: float,
    posterior_samples: int,
    posterior_seed: int,
) -> dict[str, Any]:
    ordinary_weights = {
        str(node_id): max(0.0, float(weight))
        for node_id, weight in weights.items()
        if str(node_id) != UNKNOWN_NODE_ID
    }
    unknown_weight = max(0.0, float(weights.get(UNKNOWN_NODE_ID, 0.0) or 0.0))
    total = unknown_weight + sum(ordinary_weights.values())
    if total > 0:
        unknown_weight /= total
        ordinary_weights = {
            node_id: weight / total for node_id, weight in ordinary_weights.items()
        }
    memory_support = sum(ordinary_weights.values())
    if memory_support <= 0:
        return {
            "memory_support": 0.0,
            "unknown_weight": 1.0 if total <= 0 else unknown_weight,
            "conditional_weights": {},
            "retrieval_uncertainty": 0.0,
            "tree_uncertainty": 0.0,
            "memory_commitment": 0.0,
            "memory_competence": None,
            "reference_competence": _posterior_mean(unknown),
            "competence_below_reference_probability": 0.0,
            "overconfidence_risk": 0.0,
        }

    conditional = {
        node_id: weight / memory_support for node_id, weight in ordinary_weights.items()
    }
    if len(conditional) <= 1:
        retrieval_uncertainty = 0.0
    else:
        entropy = -sum(
            weight * math.log(weight)
            for weight in conditional.values()
            if weight > 0
        )
        retrieval_uncertainty = _clip(entropy / math.log(len(conditional)))

    tree_uncertainty = 0.0
    for left_id, left_weight in conditional.items():
        for right_id, right_weight in conditional.items():
            tree_uncertainty += (
                left_weight
                * right_weight
                * normalized_tree_distance(left_id, right_id, nodes)
            )
    tree_uncertainty = _clip(tree_uncertainty)
    commitment = _clip(memory_support * (1.0 - tree_uncertainty))

    nodes_by_id = {
        str(node.get("node_id", "")).strip(): node
        for node in nodes
        if str(node.get("node_id", "")).strip()
    }
    competence = sum(
        conditional[node_id] * _posterior_mean(nodes_by_id.get(node_id, {}))
        for node_id in conditional
    )
    reference = _posterior_mean(unknown)
    rng = random.Random(posterior_seed)
    below_count = 0
    sample_count = max(1, int(posterior_samples))
    for _ in range(sample_count):
        aggregated = sum(
            conditional[node_id]
            * rng.betavariate(*_posterior_parameters(nodes_by_id.get(node_id, {})))
            for node_id in conditional
        )
        reference_sample = rng.betavariate(*_posterior_parameters(unknown))
        if aggregated < reference_sample:
            below_count += 1
    below_probability = below_count / sample_count
    overconfidence = _clip(
        commitment
        * max(0.0, reference - competence)
        * below_probability
    )
    return {
        "memory_support": memory_support,
        "unknown_weight": unknown_weight,
        "conditional_weights": conditional,
        "retrieval_uncertainty": retrieval_uncertainty,
        "tree_uncertainty": tree_uncertainty,
        "memory_commitment": commitment,
        "memory_competence": competence,
        "reference_competence": reference,
        "competence_below_reference_probability": below_probability,
        "overconfidence_risk": overconfidence,
    }


def calibrate_support_weights(
    *,
    weights: dict[str, float],
    nodes: list[dict[str, Any]],
    metrics: dict[str, Any],
    calibration_strength: float,
    epsilon: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply the paper's node calibration in Equations (12) and (13)."""

    normalized = {
        str(node_id): max(0.0, float(weight))
        for node_id, weight in weights.items()
    }
    total = sum(normalized.values())
    if total <= 0:
        return {UNKNOWN_NODE_ID: 1.0}, {}
    normalized = {node_id: weight / total for node_id, weight in normalized.items()}
    normalized.setdefault(UNKNOWN_NODE_ID, 0.0)

    conditional = {
        str(node_id): max(0.0, float(weight))
        for node_id, weight in (metrics.get("conditional_weights", {}) or {}).items()
    }
    reference = float(metrics.get("reference_competence", 0.5) or 0.5)
    nodes_by_id = {
        str(node.get("node_id", "")).strip(): node
        for node in nodes
        if str(node.get("node_id", "")).strip()
    }
    unnormalized_deficits = {
        node_id: conditional_weight
        * max(0.0, reference - _posterior_mean(nodes_by_id.get(node_id, {})))
        for node_id, conditional_weight in conditional.items()
    }
    denominator = sum(unnormalized_deficits.values()) + max(float(epsilon), 1e-15)
    deficits = {
        node_id: value / denominator
        for node_id, value in unnormalized_deficits.items()
    }

    risk = max(0.0, float(metrics.get("overconfidence_risk", 0.0) or 0.0))
    kappa = max(0.0, float(calibration_strength))
    adjusted = {
        node_id: weight * math.exp(-kappa * risk * deficits.get(node_id, 0.0))
        for node_id, weight in normalized.items()
    }
    adjusted_total = sum(adjusted.values())
    if adjusted_total <= 0:
        return {UNKNOWN_NODE_ID: 1.0}, deficits
    return (
        {node_id: value / adjusted_total for node_id, value in adjusted.items()},
        deficits,
    )


def apply_injection_threshold(
    weights: dict[str, float],
    *,
    threshold: float,
) -> dict[str, float]:
    """Move non-injected ordinary-node mass to the unknown node."""

    cutoff = _clip(threshold)
    kept: dict[str, float] = {}
    unknown = max(0.0, float(weights.get(UNKNOWN_NODE_ID, 0.0) or 0.0))
    for node_id, weight in weights.items():
        if node_id == UNKNOWN_NODE_ID:
            continue
        value = max(0.0, float(weight))
        if value < cutoff:
            unknown += value
        else:
            kept[node_id] = value
    kept[UNKNOWN_NODE_ID] = unknown
    total = sum(kept.values())
    if total <= 0:
        return {UNKNOWN_NODE_ID: 1.0}
    return {node_id: value / total for node_id, value in kept.items()}


def _posterior_parameters(value: dict[str, Any]) -> tuple[float, float]:
    prior_alpha = max(1e-6, float(value.get("prior_alpha", 1.0) or 1.0))
    prior_beta = max(1e-6, float(value.get("prior_beta", 1.0) or 1.0))
    success = max(0.0, float(value.get("success_count", 0.0) or 0.0))
    failure = max(0.0, float(value.get("failure_count", 0.0) or 0.0))
    return prior_alpha + success, prior_beta + failure


def _posterior_mean(value: dict[str, Any]) -> float:
    alpha, beta = _posterior_parameters(value)
    return alpha / (alpha + beta)


class C2Memory(BaseOnlineMemory):
    """Behavior-tree memory with commitment, competence, and risk estimation."""

    backend_name = "c2mem"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.commitment_lambda = _clip(_float_env("C2MEM_COMMITMENT_LAMBDA", 0.5))
        self.calibration_strength = max(
            0.0, _float_env("C2MEM_CALIBRATION_KAPPA", 1.0)
        )
        self.calibration_epsilon = max(
            1e-15, _float_env("C2MEM_CALIBRATION_EPSILON", 1e-12)
        )
        self.injection_threshold = _clip(
            _float_env("C2MEM_INJECTION_THRESHOLD", 0.0)
        )
        self.posterior_samples = max(1, _int_env("C2MEM_POSTERIOR_SAMPLES", 1000))
        self.split_min_size = max(4, _int_env("C2MEM_SPLIT_MIN_SIZE", 8))
        self.split_similarity_threshold = _clip(
            _float_env("C2MEM_SPLIT_SIMILARITY_THRESHOLD", 0.45)
        )
        self.max_events = max(1, _int_env("C2MEM_MAX_EVENTS", 1000))
        self._task: MemoryTask | None = None
        self._live_traces: list[dict[str, Any]] = []
        self._top_k = 2
        self._task_events: list[dict[str, float]] = []
        state = self.store.state
        stored_schema = str(state.get("schema_version", "")).strip()
        if stored_schema and stored_schema != C2MEM_STATE_SCHEMA:
            raise ValueError(
                f"incompatible C2Mem state schema {stored_schema!r}; "
                "paper-aligned calibration requires a new memory directory"
            )
        state["schema_version"] = C2MEM_STATE_SCHEMA
        state.setdefault("nodes", [])
        state.setdefault("record_node_ids", {})
        state.setdefault("retrieval_events", [])
        state.setdefault(
            "unknown_node",
            {
                "node_id": UNKNOWN_NODE_ID,
                "prior_alpha": _float_env("C2MEM_UNKNOWN_PRIOR_ALPHA", 1.0),
                "prior_beta": _float_env("C2MEM_UNKNOWN_PRIOR_BETA", 1.0),
                "success_count": 0.0,
                "failure_count": 0.0,
            },
        )
        state["config"] = {
            "commitment_lambda": self.commitment_lambda,
            "calibration_strength": self.calibration_strength,
            "calibration_epsilon": self.calibration_epsilon,
            "injection_threshold": self.injection_threshold,
            "posterior_samples": self.posterior_samples,
            "split_min_size": self.split_min_size,
            "split_similarity_threshold": self.split_similarity_threshold,
        }

    @property
    def _nodes(self) -> list[dict[str, Any]]:
        return self.store.state.setdefault("nodes", [])

    @property
    def _unknown(self) -> dict[str, Any]:
        return self.store.state["unknown_node"]

    async def begin_task(self, task: MemoryTask) -> None:
        self._task = task
        self._live_traces = []
        self._task_events = []

    def _execution_prefix(self, task: MemoryTask) -> str:
        explicit = str(task.metadata.get("supervisor_task", "")).strip()
        base = explicit or task.text or task.query
        live = "\n".join(
            f"[{trace.get('agent', 'agent')}] {str(trace.get('output', ''))[:1200]}"
            for trace in self._live_traces[-6:]
            if str(trace.get("output", "")).strip()
        )
        if live and live not in base:
            return f"{base[:9000]}\n\nRecent multi-agent execution:\n{live}"
        return base[:12000]

    def _leaf_nodes(self) -> list[dict[str, Any]]:
        return [node for node in self._nodes if not node.get("child_ids")]

    async def _retrieve_candidates(
        self,
        prefix: str,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        leaves = self._leaf_nodes()
        if not leaves or top_k <= 0:
            return [], {}
        query_vector = await self._embed(
            "Behavioral execution prefix; ignore task entities and topic:\n" + prefix
        )
        ranked = sorted(
            (
                (
                    node,
                    cosine(
                        query_vector,
                        [float(value) for value in node.get("embedding", []) or []],
                    ),
                )
                for node in leaves
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [node for node, _ in ranked], {
            str(node["node_id"]): similarity for node, similarity in ranked
        }

    async def _evaluate_support(
        self,
        prefix: str,
        candidates: list[dict[str, Any]],
        similarities: dict[str, float],
    ) -> tuple[dict[str, float], str]:
        if not candidates:
            return {UNKNOWN_NODE_ID: 1.0}, "no ordinary behavior nodes"
        candidate_text = "\n\n".join(
            (
                f"node_id: {node['node_id']}\n"
                f"behavior_pattern: {node.get('summary', '')}\n"
                f"historical_competence: {_posterior_mean(node):.4f}"
            )
            for node in candidates
        )
        raw = await self._chat(
            EVALUATOR_SYSTEM,
            f"Current execution prefix:\n{prefix}\n\nCandidate nodes:\n{candidate_text}",
            max_tokens=500,
        )
        parsed = _json_object(raw)
        ids = [str(node["node_id"]) for node in candidates]
        weights = _normalize_weights(
            parsed.get("weights"),
            ids,
            fallback_similarities=similarities,
        )
        return weights, str(parsed.get("rationale", "")).strip()

    def _records_for_node(self, node_id: str) -> list[MemoryRecord]:
        mapping = self.store.state.get("record_node_ids", {}) or {}
        return [
            record
            for record in self.store.records
            if str(mapping.get(record.memory_id, "")) == node_id
        ]

    @staticmethod
    def _record_rules(record: MemoryRecord) -> dict[str, str]:
        return _agent_map(record.metadata.get("reusable_experience", {}))

    def _guidance_for_node(self, node_id: str, agent_id: str) -> str:
        records = self._records_for_node(node_id)
        records.sort(key=lambda record: (record.label, record.created_index), reverse=True)
        for record in records:
            rules = self._record_rules(record)
            if agent_id in rules:
                return rules[agent_id]
            if "shared" in rules:
                return rules["shared"]
            if rules:
                return next(iter(rules.values()))
        node = next(
            (item for item in self._nodes if str(item.get("node_id")) == node_id),
            {},
        )
        rules = _agent_map(node.get("reusable_experience", {}))
        return rules.get(agent_id) or rules.get("shared") or next(iter(rules.values()), "")

    def _format_contexts(
        self,
        task: MemoryTask,
        candidates: list[dict[str, Any]],
        weights: dict[str, float],
        metrics: dict[str, Any],
    ) -> dict[str, list[str]]:
        selected = [
            node
            for node in sorted(
                candidates,
                key=lambda item: weights.get(str(item["node_id"]), 0.0),
                reverse=True,
            )
            if weights.get(str(node["node_id"]), 0.0) > 0
        ]
        contexts: dict[str, list[str]] = {}
        for agent_id in task.agent_ids:
            snippets: list[str] = []
            for index, node in enumerate(selected):
                node_id = str(node["node_id"])
                guidance = self._guidance_for_node(node_id, agent_id)
                metric_line = ""
                if index == 0:
                    competence = metrics.get("memory_competence")
                    metric_line = (
                        "\nC2 diagnostics: "
                        f"support={metrics['memory_support']:.3f}, "
                        f"commitment={metrics['memory_commitment']:.3f}, "
                        f"competence={competence:.3f}, "
                        f"overconfidence_risk={metrics['overconfidence_risk']:.3f}"
                        if competence is not None
                        else "\nC2 diagnostics: no ordinary memory competence"
                    )
                snippets.append(
                    (
                        f"[c2mem behavior memory | node={node_id} | "
                        f"calibrated_weight={weights.get(node_id, 0.0):.3f}]\n"
                        f"Behavior pattern: {str(node.get('summary', ''))[:1600]}\n"
                        f"Action rule: {guidance[:1000] or 'Use only when the current feedback matches this pattern.'}"
                        f"{metric_line}"
                    ).strip()
                )
            if snippets:
                contexts[agent_id] = snippets
        return contexts

    async def retrieve_contexts(
        self,
        task: MemoryTask,
        *,
        top_k: int,
    ) -> dict[str, list[str]]:
        self._top_k = max(0, int(top_k))
        prefix = self._execution_prefix(task)
        candidates, similarities = await self._retrieve_candidates(prefix, self._top_k)
        weights, rationale = await self._evaluate_support(prefix, candidates, similarities)
        metrics = compute_c2_metrics(
            weights=weights,
            nodes=self._nodes,
            unknown=self._unknown,
            commitment_lambda=self.commitment_lambda,
            posterior_samples=self.posterior_samples,
            posterior_seed=len(self.store.state.get("retrieval_events", [])),
        )
        calibrated_weights, deficit_contributions = calibrate_support_weights(
            weights=weights,
            nodes=self._nodes,
            metrics=metrics,
            calibration_strength=self.calibration_strength,
            epsilon=self.calibration_epsilon,
        )
        injection_weights = apply_injection_threshold(
            calibrated_weights,
            threshold=self.injection_threshold,
        )
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_index": task.metadata.get("task_index"),
            "candidate_ids": [str(node["node_id"]) for node in candidates],
            "similarities": similarities,
            "weights": weights,
            "calibrated_weights": calibrated_weights,
            "injection_weights": injection_weights,
            "deficit_contributions": deficit_contributions,
            "rationale": rationale,
            "metrics": metrics,
        }
        events = self.store.state.setdefault("retrieval_events", [])
        events.append(event)
        del events[:-self.max_events]
        self.store.state["last_retrieval"] = event
        self._task_events.append(dict(injection_weights))
        self.store.save_state()
        return self._format_contexts(task, candidates, injection_weights, metrics)

    async def observe_trace(
        self,
        trace: dict[str, Any],
        *,
        agent_ids: tuple[str, ...],
    ) -> dict[str, list[str]]:
        if str(trace.get("output", "")).strip():
            self._live_traces.append(dict(trace))
        if self._task is None:
            return {}
        live_task = MemoryTask(
            text=self._execution_prefix(self._task),
            metadata=dict(self._task.metadata),
            agent_ids=agent_ids,
        )
        return await self.retrieve_contexts(live_task, top_k=self._top_k)

    async def _summarize_episode(
        self,
        task: MemoryTask,
        outcome: MemoryOutcome,
        trajectory: str,
    ) -> dict[str, Any]:
        raw = await self._chat(
            EPISODE_SUMMARY_SYSTEM,
            (
                f"Authoritative outcome: {'success' if outcome.success else 'failure'}\n"
                f"Task:\n{task.query}\n\nTrajectory:\n{trajectory}\n\n"
                f"Feedback:\n{outcome.feedback}"
            ),
            max_tokens=900,
        )
        parsed = _json_object(raw)
        content_summary = str(parsed.get("content_summary", "")).strip() or task.query[:800]
        behavior_pattern = str(parsed.get("behavior_pattern", "")).strip()
        if not behavior_pattern:
            behavior_pattern = (
                "Trigger: A multi-agent task begins. Early: The agents attempt the task using "
                "their assigned roles. Feedback: The completed trajectory provides an "
                f"{'successful' if outcome.success else 'unsuccessful'} outcome signal. "
                "Transition: Preserve the observed decision sequence and revise any unsupported "
                "step. Stop: Commit only when the task-specific completion condition is met."
            )
        outcome_summary = str(parsed.get("outcome_summary", "")).strip()
        if not outcome_summary:
            outcome_summary = (
                "The recorded trajectory completed the task."
                if outcome.success
                else "The recorded trajectory did not complete the task."
            )
        return {
            "content_summary": content_summary,
            "behavior_pattern": behavior_pattern,
            "outcome_summary": outcome_summary,
            "reusable_experience": _agent_map(parsed.get("reusable_experience", {})),
        }

    def _new_node(
        self,
        *,
        summary: str,
        embedding: list[float],
        memory_ids: list[str],
        parent_id: str = "",
        reusable_experience: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        next_index = int(self.store.state.get("next_node_index", 1) or 1)
        self.store.state["next_node_index"] = next_index + 1
        return {
            "node_id": f"n{next_index}",
            "parent_id": parent_id,
            "child_ids": [],
            "depth": (
                0
                if not parent_id
                else 1
                + int(
                    next(
                        (
                            node.get("depth", 0)
                            for node in self._nodes
                            if str(node.get("node_id")) == parent_id
                        ),
                        0,
                    )
                )
            ),
            "summary": summary,
            "embedding": list(embedding),
            "memory_ids": list(memory_ids),
            "reusable_experience": dict(reusable_experience or {}),
            "prior_alpha": _float_env("C2MEM_NODE_PRIOR_ALPHA", 1.0),
            "prior_beta": _float_env("C2MEM_NODE_PRIOR_BETA", 1.0),
            "success_count": 0.0,
            "failure_count": 0.0,
            "created_index": next_index,
        }

    async def _summarize_node(self, records: list[MemoryRecord]) -> tuple[str, dict[str, str]]:
        patterns = "\n".join(
            f"- {record.key_steps}" for record in records if record.key_steps
        )
        raw = await self._chat(
            NODE_SUMMARY_SYSTEM,
            f"Member behavior patterns:\n{patterns[:22000]}",
            max_tokens=700,
        )
        parsed = _json_object(raw)
        summary = str(parsed.get("behavior_pattern", "")).strip()
        if not summary:
            summary = next((record.key_steps for record in records if record.key_steps), "")
        return summary, _agent_map(parsed.get("reusable_experience", {}))

    @staticmethod
    def _centroid(records: list[MemoryRecord]) -> list[float]:
        vectors = [record.embedding for record in records if record.embedding]
        if not vectors:
            return []
        width = len(vectors[0])
        compatible = [vector for vector in vectors if len(vector) == width]
        return [
            sum(vector[index] for vector in compatible) / len(compatible)
            for index in range(width)
        ]

    def _member_outcomes(self, records: list[MemoryRecord]) -> tuple[float, float]:
        return (
            float(sum(1 for record in records if record.label)),
            float(sum(1 for record in records if not record.label)),
        )

    async def _refresh_node(self, node: dict[str, Any]) -> None:
        records = self._records_for_node(str(node["node_id"]))
        summary, rules = await self._summarize_node(records)
        node["summary"] = summary
        node["reusable_experience"] = rules
        node["embedding"] = await self._embed(
            "Reusable ordered behavior pattern:\n" + summary
        )
        success, failure = self._member_outcomes(records)
        node["member_success_count"] = success
        node["member_failure_count"] = failure

    def _node_heterogeneity(self, records: list[MemoryRecord]) -> float:
        similarities = [
            cosine(records[left].embedding, records[right].embedding)
            for left in range(len(records))
            for right in range(left + 1, len(records))
            if records[left].embedding and records[right].embedding
        ]
        return min(similarities, default=1.0)

    @staticmethod
    def _partition_records(records: list[MemoryRecord]) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
        farthest: tuple[int, int] | None = None
        farthest_similarity = math.inf
        for left in range(len(records)):
            for right in range(left + 1, len(records)):
                similarity = cosine(records[left].embedding, records[right].embedding)
                if similarity < farthest_similarity:
                    farthest_similarity = similarity
                    farthest = (left, right)
        if farthest is None:
            return records, []
        left_seed = records[farthest[0]].embedding
        right_seed = records[farthest[1]].embedding
        left_group: list[MemoryRecord] = []
        right_group: list[MemoryRecord] = []
        for record in records:
            if cosine(record.embedding, left_seed) >= cosine(record.embedding, right_seed):
                left_group.append(record)
            else:
                right_group.append(record)
        return left_group, right_group

    async def _maybe_split(self, node: dict[str, Any]) -> None:
        records = self._records_for_node(str(node["node_id"]))
        if len(records) < self.split_min_size:
            return
        if self._node_heterogeneity(records) >= self.split_similarity_threshold:
            return
        left_records, right_records = self._partition_records(records)
        if not left_records or not right_records:
            return
        children: list[dict[str, Any]] = []
        mapping = self.store.state.setdefault("record_node_ids", {})
        parent_success = max(0.0, float(node.get("success_count", 0.0) or 0.0))
        parent_failure = max(0.0, float(node.get("failure_count", 0.0) or 0.0))
        for group in (left_records, right_records):
            summary, rules = await self._summarize_node(group)
            child = self._new_node(
                summary=summary,
                embedding=await self._embed("Reusable ordered behavior pattern:\n" + summary),
                memory_ids=[record.memory_id for record in group],
                parent_id=str(node["node_id"]),
                reusable_experience=rules,
            )
            success, failure = self._member_outcomes(group)
            evidence_share = len(group) / len(records)
            child["success_count"] = parent_success * evidence_share
            child["failure_count"] = parent_failure * evidence_share
            child["member_success_count"] = success
            child["member_failure_count"] = failure
            children.append(child)
            for record in group:
                mapping[record.memory_id] = child["node_id"]
        node["child_ids"] = [child["node_id"] for child in children]
        self._nodes.extend(children)

    async def _insert_record(
        self,
        record: MemoryRecord,
        *,
        exposure: dict[str, float],
    ) -> str:
        leaves_by_id = {
            str(node.get("node_id", "")): node for node in self._leaf_nodes()
        }
        weighted_leaves = [
            (node, max(0.0, float(exposure.get(node_id, 0.0) or 0.0)))
            for node_id, node in leaves_by_id.items()
            if max(0.0, float(exposure.get(node_id, 0.0) or 0.0)) > 0.0
        ]
        selected = (
            max(
                weighted_leaves,
                key=lambda item: (
                    item[1],
                    -int(item[0].get("created_index", 0) or 0),
                ),
            )[0]
            if weighted_leaves
            else None
        )
        if selected is None:
            selected = self._new_node(
                summary=record.key_steps,
                embedding=record.embedding,
                memory_ids=[record.memory_id],
                reusable_experience=self._record_rules(record),
            )
            self._nodes.append(selected)
        else:
            selected.setdefault("memory_ids", []).append(record.memory_id)
        self.store.state.setdefault("record_node_ids", {})[record.memory_id] = selected["node_id"]
        await self._refresh_node(selected)
        await self._maybe_split(selected)
        return str(self.store.state["record_node_ids"][record.memory_id])

    def _task_exposure(self) -> dict[str, float]:
        accumulated: dict[str, float] = {}
        for weights in self._task_events:
            for node_id, weight in weights.items():
                accumulated[node_id] = accumulated.get(node_id, 0.0) + max(
                    0.0, float(weight)
                )
        exposure_total = sum(accumulated.values())
        if exposure_total <= 0:
            return {}
        return {
            node_id: weight / exposure_total
            for node_id, weight in accumulated.items()
        }

    def _update_retrieval_evidence(self, success: bool) -> dict[str, float]:
        exposure = self._task_exposure()
        self.store.state["last_task_exposure"] = exposure
        if not exposure:
            return exposure
        nodes_by_id = {
            str(node.get("node_id", "")): node for node in self._nodes
        }
        field = "success_count" if success else "failure_count"
        for node_id, weight in exposure.items():
            target = self._unknown if node_id == UNKNOWN_NODE_ID else nodes_by_id.get(node_id)
            if target is None:
                continue
            target[field] = float(target.get(field, 0.0) or 0.0) + weight
        return exposure

    async def record_task(self, task: MemoryTask, outcome: MemoryOutcome) -> None:
        trajectory = render_trajectory(outcome)
        summary = await self._summarize_episode(task, outcome, trajectory)
        record = MemoryRecord(
            memory_id=f"{self.backend_name}-{uuid.uuid4().hex}",
            task=task.query,
            label=bool(outcome.success),
            trajectory=trajectory,
            feedback=outcome.feedback,
            embedding_text=summary["behavior_pattern"],
            task_summary=summary["content_summary"],
            key_steps=summary["behavior_pattern"],
            failure_reason=summary["outcome_summary"],
            metadata={
                "backend": self.backend_name,
                "dataset": task.metadata.get("dataset"),
                "task_index": task.metadata.get("task_index"),
                "behavior_pattern": summary["behavior_pattern"],
                "content_summary": summary["content_summary"],
                "outcome_summary": summary["outcome_summary"],
                "reusable_experience": summary["reusable_experience"],
            },
            created_index=len(self.store.records),
        )
        record.embedding = await self._embed(
            "Reusable ordered behavior pattern:\n" + record.embedding_text
        )
        exposure = self._update_retrieval_evidence(outcome.success)
        self.store.append(record)
        node_id = await self._insert_record(record, exposure=exposure)
        record.metadata["c2mem_node_id"] = node_id
        record.metadata["c2mem_insertion_exposure"] = dict(exposure)
        self.store.save_records()
        self.store.state["last_inserted_memory_id"] = record.memory_id
        self.store.state["last_inserted_node_id"] = node_id
        self.store.save_state()
        self._task = None
        self._live_traces = []
        self._task_events = []

    def _format_record(self, record: MemoryRecord, score: float, agent_id: str) -> str:
        return (
            f"[c2mem episode | {'success' if record.label else 'failure'}]\n"
            f"content: {record.task_summary[:700]}\n"
            f"behavior: {record.key_steps[:1400]}\n"
            f"outcome: {record.failure_reason[:600]}"
        )
