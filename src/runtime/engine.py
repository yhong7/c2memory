from collections import Counter
import asyncio
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
import math
import re
import time

import numpy as np

from src.agents import BaseAgent, FinalAgent, SubAgentCoreAgent, SupervisorAgent
from src.api import GPTChat, observe, propagate_trace_attributes
from src.configs import load_config
from src.core.types import DyLANState, MASState, TaskInput
from src.log import logger

def _edbg(msg: str) -> None:
    logger.debug(msg)


def _persist_mas_trace(
    agent_traces: list[dict[str, Any]],
    task_input: "TaskInput",
    result: dict,
    round_idx: int,
) -> None:
    """Save full MAS internal I/O trace to temp/mas_trace_<ts>.json."""
    import json, os, uuid
    from datetime import datetime, timezone

    os.makedirs("temp", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    task_index = str(getattr(task_input, "metadata", {}).get("task_index", "task"))
    safe_task_index = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in task_index)
    path = f"temp/mas_trace_{ts}_{safe_task_index}_r{round_idx}_{uuid.uuid4().hex[:8]}.json"
    # Strip system_prompt from each trace — not needed in debug output
    clean_traces = [
        {k: v for k, v in t.items() if k != "system_prompt"}
        for t in agent_traces
    ]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": task_input.task if hasattr(task_input, "task") else str(task_input),
        "task_metadata": task_input.metadata if hasattr(task_input, "metadata") else {},
        "round": round_idx,
        "final_output": result.get("final_output", ""),
        "predicted_answer": result.get("predicted_answer", ""),
        "reward": result.get("reward"),
        "final_f1": result.get("final_f1"),
        "done": result.get("done"),
        "terminated": result.get("terminated"),
        "feedback": result.get("feedback", ""),
        "steps_taken": result.get("steps_taken"),
        "supervisor_brief": result.get("supervisor_brief", ""),
        "agent_traces": clean_traces,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _edbg(f"[ENGINE] mas trace saved to {path}")
from src.graphs import build_graph
from src.memory import EmptyMemoryRetriever, EmptyMemoryUpdater
from src.memory.native_mas_memory import NativeMASMemoryStore
from src.runtime.prompts import (
    DYLAN_DECISION_PROMPT,
    DYLAN_HOTPOTQA_AGGREGATOR_PROMPT,
    DYLAN_HOTPOTQA_DECISION_PROMPT,
    DYLAN_HOTPOTQA_GROUND_TRUTH_PROMPT,
    DYLAN_FEVER_SOLVER_PROMPT,
    DYLAN_HOTPOTQA_SOLVER_PROMPT,
    DYLAN_RANKER_PROMPT,
    DYLAN_ROLE_MAP,
    DYLAN_SOLVER_PROMPT,
    DYLAN_VALID_ROLES,
    FEVER_FINAL_PROMPT,
    FINAL_PROMPT,
    HOTPOTQA_ACTION_SCHEMA_NOTE,
    HOTPOTQA_FINAL_PROMPT,
    LLM_DEBATE_ROLE_PROMPTS,
    LLM_DEBATE_VALID_ROLES,
    SUB_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
    build_dylan_decision_prompt,
    build_dylan_final_fever_prompt,
    build_dylan_final_hotpotqa_prompt,
    build_dylan_final_prompt,
    build_dylan_layer_prompt,
    build_dylan_ranker_prompt,
    build_final_fever_prompt,
    build_final_hotpotqa_prompt,
    build_final_prompt,
    build_llm_debate_initial_prompt,
    build_llm_debate_judge_prompt,
    build_llm_debate_round_prompt,
    build_supervisor_prompt,
)
from src.runtime.agent_node import AgentNodeBase
from src.runtime.subagent import SubAgentRuntime
from src.runtime.tracer import RoundRecord, TokenEvent, TraceRecorder


# ---------------------------------------------------------------------------
# DCE – Deep Copy Everything (LangGraph state isolation)
# ---------------------------------------------------------------------------

def dce_copy(state: dict) -> dict:
    """Return a deep copy of *state*, skipping any unpicklable values.

    This is the safe DCE (Deep Copy Everything) utility: deep copies each
    key-value pair independently so that a single unpicklable entry (e.g. an
    internal LangGraph runtime object) does not fail the whole copy.
    """
    safe: dict = {}
    for key, value in state.items():
        try:
            safe[key] = deepcopy(value)
        except (TypeError, AttributeError):
            safe[key] = value
    return safe


def _extract_action_line(output: str) -> str:
    """Return the executable action from a model output."""
    import re

    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith("action:"):
            candidate = line.split(":", 1)[1].strip()
            match = re.search(r"\b(Search|Lookup|Finish)\[[^\]]+\]", candidate)
            if match:
                return match.group(0).strip()
    match = re.search(r"\b(Search|Lookup|Finish)\[[^\]]+\]", output)
    if match:
        return match.group(0).strip()
    return "(no valid action emitted)"


def _extract_reason_line(output: str) -> str:
    """Return the reason line from a model output if present."""
    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith("reason:"):
            return line.split(":", 1)[1].strip()
    return ""


def _extract_search_lookup_finish_action(output: str) -> str:
    """Backward-compatible alias for action extraction."""
    return _extract_action_line(output)


def _is_placeholder_hotpotqa_action(action: str) -> bool:
    match = re.fullmatch(r"(Search|Lookup|Finish)\[(.*)\]", str(action or "").strip())
    if not match:
        return False
    argument = match.group(2).strip().strip("<>\"'`").casefold()
    placeholders = {
        "query",
        "title",
        "answer",
        "actual query",
        "actual title",
        "actual answer",
    }
    return not argument or argument in placeholders


TIMEOUT_SENTINEL = "MAX_STEPS_EXCEEDED"

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
@dataclass
class MinimalLangGraphRuntime:
    graph_name: str = "supervisor_fanout"
    subagent_count: int = 4
    model_name: str | None = None
    api_profile_id: str | None = None
    read_only: bool = False
    mas_memory_backend: str = "none"
    mas_memory_source: str | None = None
    mas_memory_topk: int = 2

    # DyLAN config ─ ignored when graph_name != "dylan"
    dylan_max_layers: int = 5
    dylan_height: int = 3
    dylan_roles: list[str] = field(default_factory=lambda: ["solver", "ground_truth", "decision"])
    llm_debate_agent_num: int = 3
    llm_debate_round_num: int = 2
    llm_debate_roles: list[str] = field(default_factory=lambda: ["solver", "critic", "fact_checker"])
    llm_debate_use_judge: bool = True
    llm_debate_use_majority: bool = True
    _task_trace_context: dict[str, str] | None = field(default=None, init=False, repr=False)
    _agent_traces: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _agent_traces_all_attempts: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _current_fever_step: int = field(default=0, init=False, repr=False)
    _current_attempt_idx: int = field(default=0, init=False, repr=False)
    _memory_store: NativeMASMemoryStore | None = field(default=None, init=False, repr=False)
    _memory_agent_contexts: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _autogen_action_history: list[str] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    def __post_init__(self):
        config = load_config()
        explicit_model = self.model_name

        from src.api.api_manager import APIRequestManager
        self.api_manager = APIRequestManager(api_profile_id=self.api_profile_id, model_name=explicit_model)
        self._root_api_manager = self.api_manager
        selected_model = self.api_manager.model_name or config.get("api_config", {}).get("model_name", "gpt-3.5-turbo")

        self.supervisor = SupervisorAgent(
            name="supervisor",
            system_prompt=SUPERVISOR_PROMPT,
            llm=GPTChat(selected_model, api_manager=self.api_manager),
        )
        self.final_agent = FinalAgent(
            name="final",
            system_prompt=FINAL_PROMPT,
            llm=GPTChat(selected_model, api_manager=self.api_manager),
        )
        self.subagent_names: list[str] = []
        self.subagents: dict[str, SubAgentRuntime] = {}

        if self.graph_name == "dylan":
            self._init_dylan(selected_model, config)
        elif self.graph_name == "llm_debate":
            self._init_llm_debate(selected_model, config)
        elif self.graph_name == "autogen":
            self._init_autogen(selected_model)
        elif self.graph_name == "macnet":
            self._init_macnet(selected_model, config)
        else:
            self.subagent_names = [f"subagent_{idx+1}" for idx in range(self.subagent_count)]
            self.subagents = {
                name: SubAgentRuntime(
                    name=name,
                    core_agent=SubAgentCoreAgent(
                        name=name,
                        system_prompt=SUB_AGENT_PROMPT,
                        llm=GPTChat(selected_model, api_manager=self.api_manager),
                    ),
                    retriever=EmptyMemoryRetriever(),
                    updater=EmptyMemoryUpdater(),
                )
                for name in self.subagent_names
            }

        self.graph = build_graph(self.graph_name, self)
        backend = str(self.mas_memory_backend or "none").strip().lower()
        if backend != "none":
            if not self.mas_memory_source:
                raise ValueError("--mas-memory-path must name a writable backend state directory")
            self._memory_store = NativeMASMemoryStore.load(
                backend=backend,
                persist_dir=self.mas_memory_source,
            )

    def _set_api_manager(self, api_manager) -> None:
        self.api_manager = api_manager
        for agent in (
            getattr(self, "supervisor", None),
            getattr(self, "final_agent", None),
            getattr(self, "_dylan_decision", None),
            getattr(self, "_dylan_ranker", None),
        ):
            if agent is not None and getattr(agent, "llm", None) is not None:
                agent.llm.api_manager = api_manager
        for agent in getattr(self, "_dylan_agents", []) or []:
            if getattr(agent, "llm", None) is not None:
                agent.llm.api_manager = api_manager
        for agent in (
            getattr(self, "_autogen_solver", None),
            getattr(self, "_autogen_recovery", None),
            getattr(self, "_macnet_decision", None),
            getattr(self, "_llm_debate_judge", None),
        ):
            if agent is not None and getattr(agent, "llm", None) is not None:
                agent.llm.api_manager = api_manager
        for agent in getattr(self, "_macnet_agents", []) or []:
            if getattr(agent, "llm", None) is not None:
                agent.llm.api_manager = api_manager
        for agent in getattr(self, "_llm_debate_agents", []) or []:
            if getattr(agent, "llm", None) is not None:
                agent.llm.api_manager = api_manager
        for subagent in getattr(self, "subagents", {}).values():
            core_agent = getattr(subagent, "core_agent", None)
            if core_agent is not None and getattr(core_agent, "llm", None) is not None:
                core_agent.llm.api_manager = api_manager
        embed_fn = getattr(self, "_embed_fn", None)
        if embed_fn is not None and hasattr(embed_fn, "_api_manager"):
            embed_fn._api_manager = api_manager

    def _begin_task_api_scope(self, metadata: dict[str, Any]) -> None:
        root_manager = getattr(self, "_root_api_manager", self.api_manager)
        dataset = str(metadata.get("dataset", "unknown"))
        task_index = str(metadata.get("task_index", "task"))
        task_manager = root_manager.raise_child(
            f"task:{dataset}:{task_index}",
            new_task_scope=True,
        )
        self._set_api_manager(task_manager)

    @staticmethod
    def _is_fever_task(task: TaskInput) -> bool:
        return task.metadata.get("dataset") == "fever"

    @staticmethod
    def _is_hotpotqa_task(task: TaskInput) -> bool:
        return task.metadata.get("dataset") == "hotpotqa"

    def _observe(self, name: str, **kwargs):
        trace_context = kwargs.pop("trace_context", None) or self._task_trace_context
        return observe(name, trace_context=trace_context, **kwargs)

    def _get_memory_context(self, agent_id: str) -> list[str]:
        return list(self._memory_agent_contexts.get(agent_id, []))

    def _memory_agent_ids(self) -> list[str]:
        ids = [self.supervisor.name, self.final_agent.name]
        if self.graph_name == "dylan":
            ids.extend(agent.name for agent in getattr(self, "_dylan_agents", []) or [])
            ids.append(getattr(self, "_dylan_decision", self.final_agent).name)
        elif self.graph_name == "macnet":
            ids.extend(agent.name for agent in getattr(self, "_macnet_agents", []) or [])
            ids.append(getattr(self, "_macnet_decision", self.final_agent).name)
        elif self.graph_name == "llm_debate":
            ids.extend(agent.name for agent in getattr(self, "_llm_debate_agents", []) or [])
            ids.append(getattr(self, "_llm_debate_judge", self.final_agent).name)
        elif self.graph_name == "autogen":
            for agent in (getattr(self, "_autogen_solver", None), getattr(self, "_autogen_recovery", None)):
                if agent is not None:
                    ids.append(agent.name)
        else:
            ids.extend(getattr(self, "subagent_names", []) or [])
        unique: list[str] = []
        for agent_id in ids:
            agent_id = str(agent_id or "").strip()
            if agent_id and agent_id not in unique:
                unique.append(agent_id)
        return unique

    async def _retrieve_memory_contexts(self, task_input: TaskInput) -> dict[str, list[str]]:
        if self._memory_store is None:
            return {}
        contexts = await self._memory_store.retrieve_contexts(
            task_input,
            agent_ids=self._memory_agent_ids(),
            api_manager=self.api_manager,
            top_k=max(0, int(self.mas_memory_topk or 0)),
        )
        return contexts

    async def _append_agent_trace(self, trace: dict[str, Any]) -> None:
        self._agent_traces.append(trace)
        if self._memory_store is None:
            return
        contexts = await self._memory_store.observe_trace(
            trace,
            agent_ids=self._memory_agent_ids(),
            api_manager=self.api_manager,
        )
        if contexts:
            self._memory_agent_contexts = contexts

    def _token_usage_snapshot(self) -> dict[str, int]:
        api_manager = getattr(self, "api_manager", None)
        if api_manager is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}
        return api_manager.get_token_usage()

    def _token_usage_delta(self, start: dict[str, int]) -> dict[str, int]:
        api_manager = getattr(self, "api_manager", None)
        if api_manager is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}
        current = api_manager.get_token_usage()
        return {
            "prompt_tokens": max(0, int(current.get("prompt_tokens", 0)) - int(start.get("prompt_tokens", 0))),
            "completion_tokens": max(0, int(current.get("completion_tokens", 0)) - int(start.get("completion_tokens", 0))),
        }

    def _reconcile_token_usage(self, recorder: TraceRecorder) -> None:
        api_manager = getattr(self, "api_manager", None)
        if api_manager is None or not hasattr(recorder, "record_token_event"):
            return
        usage = api_manager.get_token_usage()
        prompt_delta = max(0, int(usage.get("prompt_tokens", 0)) - int(recorder.total_prompt_tokens))
        completion_delta = max(0, int(usage.get("completion_tokens", 0)) - int(recorder.total_completion_tokens))
        recorder.record_token_event(
            TokenEvent(
                name="runtime.untracked-token-usage",
                prompt_tokens=prompt_delta,
                completion_tokens=completion_delta,
                metadata={"reason": "token usage was observed on the shared API manager but not attributed to a round/event"},
            )
        )

    @staticmethod
    def _pre_evaluation_trajectory(
        trajectory: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        sanitized = [
            {
                "action": str(turn.get("action", "")),
                "observation": str(turn.get("observation", "")),
                "raw_output": str(turn.get("raw_output", "")),
            }
            for turn in trajectory
            if isinstance(turn, dict)
        ]
        if sanitized and sanitized[-1]["action"].strip().startswith("Finish["):
            sanitized[-1]["observation"] = "(evaluation withheld)"
        return sanitized

    async def _finalize_interactive_attempt(
        self,
        task_input: TaskInput,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate the terminal action after an interactive attempt."""
        trajectory = self._pre_evaluation_trajectory(
            list(result.get("trajectory", []) or [])
        )
        pending_answer = result.pop("_pending_finish_answer", None)
        if self._is_hotpotqa_task(task_input):
            from src.eval.hotpotqa_env import HotpotQAEnv

            if pending_answer is None:
                result.update(
                    {
                        "final_output": TIMEOUT_SENTINEL,
                        "predicted_answer": TIMEOUT_SENTINEL,
                        "reward": 0.0,
                        "final_f1": 0.0,
                        "done": False,
                        "terminated": False,
                        "feedback": "You failed the task.",
                    }
                )
                return result
            env = HotpotQAEnv(
                answer=str(task_input.metadata.get("answer", "")),
                max_steps=max(1, len(trajectory)),
            )
            observation, _, _ = env.evaluate_finish(str(pending_answer))
            if trajectory:
                trajectory[-1]["observation"] = observation
            reward, done, feedback = env.feedback()
            result.update(
                {
                    "final_output": env.final_answer or "",
                    "predicted_answer": env.final_answer or "",
                    "reward": reward,
                    "final_f1": float(env.final_f1),
                    "done": done,
                    "terminated": True,
                    "feedback": feedback,
                    "trajectory": trajectory,
                }
            )
            return result

        if self._is_fever_task(task_input):
            from src.eval.fever import FeverEnv

            if pending_answer is None:
                result.update(
                    {
                        "final_output": TIMEOUT_SENTINEL,
                        "predicted_label": TIMEOUT_SENTINEL,
                        "reward": 0.0,
                        "done": False,
                        "terminated": False,
                        "feedback": "You failed the task.",
                    }
                )
                return result
            env = FeverEnv(
                answer=str(task_input.metadata.get("label", "")),
                max_steps=max(1, len(trajectory)),
            )
            observation, _, _ = env.evaluate_finish(str(pending_answer))
            if trajectory:
                trajectory[-1]["observation"] = observation
            reward, done, feedback = env.feedback()
            result.update(
                {
                    "final_output": env.final_answer or "",
                    "predicted_label": env.final_answer or "",
                    "reward": reward,
                    "done": done,
                    "terminated": True,
                    "feedback": feedback,
                    "trajectory": trajectory,
                }
            )
        return result


    @staticmethod
    def _build_trace_name(task: TaskInput, round_idx: int) -> str:
        dataset = str(task.metadata.get("dataset", "task"))
        task_index = task.metadata.get("task_index")
        if task_index is not None:
            base = f"{dataset}-{task_index}"
        else:
            base = dataset
        return base if round_idx == 0 else f"{base}-round-{round_idx}"

    @staticmethod
    def _build_attempt_trace_name(task: TaskInput, attempt_idx: int) -> str:
        return MinimalLangGraphRuntime._build_trace_name(task, attempt_idx)

    def _is_task_success(self, task_input: TaskInput, result: dict[str, Any]) -> bool:
        if self._is_fever_task(task_input) or self._is_hotpotqa_task(task_input):
            return bool(result.get("done", False))
        return len(str(result.get("final_output", "")).strip()) > 0

    async def _execute_task_attempt(
        self,
        *,
        task_input: TaskInput,
        max_steps: int,
        recorder: TraceRecorder,
        attempt_idx: int,
    ) -> dict[str, Any]:
        start_usage = self._token_usage_snapshot()
        t0 = time.perf_counter()
        self._agent_traces = []
        self._current_attempt_idx = attempt_idx
        if attempt_idx == 0:
            self._agent_traces_all_attempts = []
        self._memory_agent_contexts = {}
        self._reset_dylan_task_state()

        trace_name = self._build_attempt_trace_name(task_input, attempt_idx)
        with propagate_trace_attributes(trace_name=trace_name):
            with self._observe(
                "task-run",
                input={"task": task_input.task, "graph": self.graph_name, "attempt": attempt_idx},
                metadata={
                    "runtime": "MinimalLangGraphRuntime",
                    "attempt": attempt_idx,
                    "dataset": task_input.metadata.get("dataset"),
                },
            ) as run_observation:
                with self._push_trace_parent(run_observation):
                    if self._is_fever_task(task_input):
                        result = await self._run_fever_episode(task_input, max_steps)
                    elif self._is_hotpotqa_task(task_input):
                        result = await self._run_hotpotqa_episode(task_input, max_steps)
                    else:
                        result = await self._run_general_task(task_input)
                    if self._is_fever_task(task_input) or self._is_hotpotqa_task(task_input):
                        result = await self._finalize_interactive_attempt(
                            task_input,
                            result,
                        )
                    if run_observation is not None:
                        run_observation.update(
                            output={
                                "supervisor_brief": result.get("supervisor_brief", ""),
                                "final_output": result.get("final_output", ""),
                            }
                        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        usage = self._token_usage_delta(start_usage)
        recorder.record_round(RoundRecord(
            round_idx=attempt_idx,
            task_input=task_input.task,
            final_output=result.get("final_output", ""),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            duration_ms=elapsed_ms,
            success=self._is_task_success(task_input, result),
            predicted_answer=str(
                result.get("predicted_answer") or result.get("final_output", "")
            ),
            done=bool(result.get("done", False)),
            terminated=bool(result.get("terminated", False)),
            feedback=str(result.get("feedback", "")),
            steps_taken=result.get("steps_taken"),
            trajectory=[
                dict(turn)
                for turn in (result.get("trajectory", []) or [])
                if isinstance(turn, dict)
            ],
            metadata={
                "dataset": task_input.metadata.get("dataset", "unknown"),
            },
        ))
        _persist_mas_trace(self._agent_traces, task_input, result, attempt_idx)
        for trace in self._agent_traces:
            trace_copy = dict(trace)
            trace_copy["attempt"] = attempt_idx
            self._agent_traces_all_attempts.append(trace_copy)
        return result


    @contextmanager
    def _push_trace_parent(self, observation):
        previous_trace_context = self._task_trace_context
        if observation is not None:
            self._task_trace_context = {
                "trace_id": getattr(observation, "trace_id", ""),
                "parent_span_id": getattr(observation, "id", ""),
            }
        try:
            yield
        finally:
            self._task_trace_context = previous_trace_context

    # ==================================================================
    #  AutoGen / MacNet helpers
    # ==================================================================

    def _init_autogen(self, model: str) -> None:
        self._autogen_solver = BaseAgent(
            name="solver",
            system_prompt=(
                "You are a smart agent designed to solve problems. Work directly "
                "toward the answer. For retrieval-style tasks, identify the exact "
                "answer span and avoid adding unsupported explanation."
            ),
            llm=GPTChat(model, api_manager=self.api_manager),
        )
        self._autogen_recovery = BaseAgent(
            name="ground_truth",
            system_prompt=(
                "You are a recovery agent. You are called only when the solver "
                "appears stuck or repetitive. Analyze the task from a different "
                "angle and provide a corrected answer or concrete guidance."
            ),
            llm=GPTChat(model, api_manager=self.api_manager),
        )

    def _init_macnet(self, model: str, config: dict) -> None:
        node_num = int(config.get("macnet", {}).get("node_num", self.dylan_height) or self.dylan_height)
        node_num = max(1, node_num)
        self._macnet_agents = [
            BaseAgent(
                name=f"solver_{idx}",
                system_prompt=(
                    "You are a smart agent designed to solve problems. Use the "
                    "task, supervisor brief, and upstream agent outputs to produce "
                    "your best answer. Keep the output concise and answer-focused."
                ),
                llm=GPTChat(model, api_manager=self.api_manager),
            )
            for idx in range(node_num)
        ]
        self._macnet_decision = BaseAgent(
            name="macnet_decision",
            system_prompt=(
                "You are a decision agent. Given multiple solver outputs, synthesize "
                "one final candidate answer. Prefer evidence-backed exact answers."
            ),
            llm=GPTChat(model, api_manager=self.api_manager),
        )

    def _init_llm_debate(self, model: str, config: dict) -> None:
        cfg = dict(config.get("llm_debate", {}) or {})
        self.llm_debate_agent_num = int(cfg.get("agent_num", self.llm_debate_agent_num) or self.llm_debate_agent_num)
        self.llm_debate_round_num = int(cfg.get("round_num", self.llm_debate_round_num) or self.llm_debate_round_num)
        self.llm_debate_use_judge = bool(cfg.get("use_judge", self.llm_debate_use_judge))
        self.llm_debate_use_majority = bool(cfg.get("use_majority", self.llm_debate_use_majority))
        roles = cfg.get("roles", self.llm_debate_roles)
        roles = [str(role) for role in roles if str(role) in LLM_DEBATE_VALID_ROLES and str(role) != "judge"]
        if not roles:
            roles = ["debater"]
        self.llm_debate_roles = roles
        self.llm_debate_agent_num = max(1, self.llm_debate_agent_num)
        self.llm_debate_round_num = max(1, self.llm_debate_round_num)

        self._llm_debate_agents: list[BaseAgent] = []
        self._llm_debate_roles_by_agent: dict[str, str] = {}
        for idx in range(self.llm_debate_agent_num):
            role = roles[idx % len(roles)]
            name = f"{role}_{idx}"
            agent = BaseAgent(
                name=name,
                system_prompt=LLM_DEBATE_ROLE_PROMPTS.get(role, LLM_DEBATE_ROLE_PROMPTS["debater"]),
                llm=GPTChat(model, api_manager=self.api_manager),
            )
            self._llm_debate_agents.append(agent)
            self._llm_debate_roles_by_agent[name] = role

        self._llm_debate_judge = BaseAgent(
            name="debate_judge",
            system_prompt=LLM_DEBATE_ROLE_PROMPTS["judge"],
            llm=GPTChat(model, api_manager=self.api_manager),
        )

    async def llm_debate_node(self, state: MASState) -> dict:
        state = dce_copy(state)
        task = state["task"].task
        brief = state.get("supervisor_brief", "")
        latest_answers: dict[str, str] = {}
        outputs: dict[str, str] = {}

        async def run_agent_round(agent: BaseAgent, round_idx: int) -> tuple[str, str]:
            role = self._llm_debate_roles_by_agent.get(agent.name, "debater")
            memory_context = self._get_memory_context(agent.name)
            if round_idx == 0:
                prompt = build_llm_debate_initial_prompt(
                    task=task,
                    supervisor_brief=brief,
                    memory_context=[],
                    behavior_memory_context=memory_context,
                )
            else:
                other_answers = [
                    (
                        other.name,
                        self._llm_debate_roles_by_agent.get(other.name, "debater"),
                        latest_answers.get(other.name, ""),
                    )
                    for other in self._llm_debate_agents
                    if other.name != agent.name and latest_answers.get(other.name)
                ]
                prompt = build_llm_debate_round_prompt(
                    task=task,
                    supervisor_brief=brief,
                    self_previous_answer=latest_answers.get(agent.name, ""),
                    other_answers=other_answers,
                    round_index=round_idx,
                    memory_context=[],
                    behavior_memory_context=memory_context,
                )
            with self._observe(
                "llm-debate-agent",
                input={"task": task, "agent": agent.name, "round": round_idx, "prompt": prompt},
                metadata={"agent": agent.name, "round": round_idx, "role": role},
            ) as obs:
                output = str(await agent.respond(prompt)).strip()
                await self._append_agent_trace(
                    {
                        "step": self._current_fever_step,
                        "agent": agent.name,
                        "role": f"LLM-Debate {role} round {round_idx}",
                        "system_prompt": agent.system_prompt,
                        "input": prompt,
                        "output": output,
                        "tools": ["respond"],
                    }
                )
                if obs is not None:
                    obs.update(output={"output": output, "action": _extract_action_line(output)})
                return agent.name, output

        for round_idx in range(self.llm_debate_round_num):
            results = await asyncio.gather(
                *(run_agent_round(agent, round_idx) for agent in self._llm_debate_agents),
            )
            next_answers: dict[str, str] = {}
            for name, output in results:
                next_answers[name] = output
            latest_answers = next_answers
            outputs.update({name: value for name, value in latest_answers.items() if value})

        final_answers = [
            (
                agent.name,
                self._llm_debate_roles_by_agent.get(agent.name, "debater"),
                latest_answers.get(agent.name, ""),
            )
            for agent in self._llm_debate_agents
            if latest_answers.get(agent.name)
        ]
        final_candidate = self._llm_debate_majority_answer(final_answers) if self.llm_debate_use_majority else None

        if final_candidate is None and self.llm_debate_use_judge and final_answers:
            prompt = build_llm_debate_judge_prompt(
                task=task,
                supervisor_brief=brief,
                final_answers=final_answers,
                behavior_memory_context=self._get_memory_context(self._llm_debate_judge.name),
            )
            with self._observe(
                "llm-debate-judge",
                input={"task": task, "prompt": prompt},
                metadata={"agent": self._llm_debate_judge.name},
            ) as obs:
                final_candidate = str(await self._llm_debate_judge.respond(prompt)).strip()
                await self._append_agent_trace(
                    {
                        "step": self._current_fever_step,
                        "agent": self._llm_debate_judge.name,
                        "role": "LLM-Debate judge - selects the final debate action",
                        "system_prompt": self._llm_debate_judge.system_prompt,
                        "input": prompt,
                        "output": final_candidate,
                        "tools": ["respond"],
                    }
                )
                if obs is not None:
                    obs.update(output={"output": final_candidate, "action": _extract_action_line(final_candidate)})

        if final_candidate is None and final_answers:
            final_candidate = final_answers[0][2] if final_answers else "No debate response."
        if not final_candidate:
            raise RuntimeError("LLM-Debate produced no final candidate answer")
        outputs["llm_debate_decision"] = final_candidate
        return {"subagent_outputs": outputs}

    @staticmethod
    def _llm_debate_majority_answer(answers: list[tuple[str, str, str]]) -> str | None:
        if not answers:
            return None
        keys = [_extract_action_line(answer) for _, _, answer in answers]
        keys = [key for key in keys if key and key != "(no valid action emitted)"]
        if not keys:
            return None
        counter = Counter(keys)
        key, count = counter.most_common(1)[0]
        if count < math.ceil(len(answers) / 2):
            return None
        for _, _, answer in answers:
            if _extract_action_line(answer) == key:
                return answer
        return None

    async def autogen_solver_node(self, state: MASState) -> dict:
        state = dce_copy(state)
        task = state["task"].task
        brief = state.get("supervisor_brief", "")
        prompt = "\n\n".join(
            [
                f"Task:\n{task}",
                f"Supervisor brief:\n{brief}",
                self._format_memory_context_block(self._get_memory_context("solver")),
                "Return your best answer.",
            ]
        )
        with self._observe(
            "autogen-solver",
            input={"task": task, "prompt": prompt},
            metadata={"agent": self._autogen_solver.name},
        ) as obs:
            output = str(await self._autogen_solver.respond(prompt)).strip()
            await self._append_agent_trace(
                {
                    "step": self._current_fever_step,
                    "agent": self._autogen_solver.name,
                    "role": "AutoGen solver - primary problem solver",
                    "system_prompt": self._autogen_solver.system_prompt,
                    "input": prompt,
                    "output": output,
                    "tools": ["respond"],
                }
            )
            if obs is not None:
                obs.update(output={"output": output})
            history = list(getattr(self, "_autogen_action_history", []))
            history.append(_extract_action_line(output))
            self._autogen_action_history = history[-3:]
            return {
                "subagent_outputs": {self._autogen_solver.name: output},
                "autogen_action_history": list(self._autogen_action_history),
            }

    def autogen_router(self, state: MASState) -> str:
        history = list(state.get("autogen_action_history", []) or [])
        if len(history) >= 3 and history[-1] == history[-2] == history[-3]:
            return "recovery"
        return "final"

    async def autogen_recovery_node(self, state: MASState) -> dict:
        state = dce_copy(state)
        task = state["task"].task
        brief = state.get("supervisor_brief", "")
        solver_output = state.get("subagent_outputs", {}).get("solver", "")
        prompt = "\n\n".join(
            [
                f"Task:\n{task}",
                f"Supervisor brief:\n{brief}",
                f"Stuck solver output:\n{solver_output}",
                self._format_memory_context_block(
                    self._get_memory_context("ground_truth")
                ),
                "Give a corrected answer or concrete guidance.",
            ]
        )
        with self._observe(
            "autogen-recovery",
            input={"task": task, "prompt": prompt},
            metadata={"agent": self._autogen_recovery.name},
        ) as obs:
            output = str(await self._autogen_recovery.respond(prompt)).strip()
            await self._append_agent_trace(
                {
                    "step": self._current_fever_step,
                    "agent": self._autogen_recovery.name,
                    "role": "AutoGen recovery - breaks repeated solver behavior",
                    "system_prompt": self._autogen_recovery.system_prompt,
                    "input": prompt,
                    "output": output,
                    "tools": ["respond"],
                }
            )
            if obs is not None:
                obs.update(output={"output": output})
            return {"subagent_outputs": {self._autogen_recovery.name: output}}

    async def macnet_layer_node(self, state: MASState) -> dict:
        state = dce_copy(state)
        task_input = state["task"]
        task = task_input.task
        dataset = str((task_input.metadata or {}).get("dataset", "")).lower()
        brief = state.get("supervisor_brief", "")
        outputs: dict[str, str] = {}
        for idx, agent in enumerate(self._macnet_agents):
            upstream = (
                "\n".join(
                    f"{name}: {value}" for name, value in outputs.items()
                )
                if outputs
                else "(none)"
            )
            prompt = "\n\n".join(
                [
                    self._format_memory_context_block(
                        self._get_memory_context(agent.name)
                    ),
                    f"Task:\n{task}",
                    f"Supervisor brief:\n{brief}",
                    f"Outputs from previous graph nodes in this round:\n{upstream}",
                    "Current environment requires one next action proposal when you are ready to act.",
                    HOTPOTQA_ACTION_SCHEMA_NOTE if dataset == "hotpotqa" else "",
                    "Return either a concise reasoning contribution or one concrete next action. "
                    "If you output an action, fill the bracket with an actual grounded value.",
                ]
            )
            with self._observe(
                "macnet-node",
                input={"task": task, "agent": agent.name, "prompt": prompt},
                metadata={"agent": agent.name, "node_index": idx},
            ) as obs:
                output = str(await agent.respond(prompt)).strip()
                outputs[agent.name] = output
                await self._append_agent_trace(
                    {
                        "step": self._current_fever_step,
                        "agent": agent.name,
                        "role": f"MacNet solver node {idx}",
                        "system_prompt": agent.system_prompt,
                        "input": prompt,
                        "output": output,
                        "tools": ["respond"],
                    }
                )
                if obs is not None:
                    obs.update(output={"output": output})
        return {"subagent_outputs": outputs}

    async def macnet_decision_node(self, state: MASState) -> dict:
        state = dce_copy(state)
        task_input = state["task"]
        task = task_input.task
        dataset = str((task_input.metadata or {}).get("dataset", "")).lower()
        brief = state.get("supervisor_brief", "")
        outputs = state.get("subagent_outputs", {}) or {}
        rendered_outputs = "\n".join(
            f"{name}: {value}" for name, value in outputs.items()
        )
        prompt = "\n\n".join(
            [
                self._format_memory_context_block(
                    self._get_memory_context(self._macnet_decision.name)
                ),
                f"Task:\n{task}",
                f"Supervisor brief:\n{brief}",
                f"Solver graph outputs:\n{rendered_outputs}",
                "Synthesize the graph outputs into the best next environment action candidate.",
                HOTPOTQA_ACTION_SCHEMA_NOTE if dataset == "hotpotqa" else "",
                "Return a concrete action when the next step is clear; bracket contents must be actual grounded values.",
            ]
        )
        with self._observe(
            "macnet-decision",
            input={"task": task, "prompt": prompt},
            metadata={"agent": self._macnet_decision.name},
        ) as obs:
            output = str(await self._macnet_decision.respond(prompt)).strip()
            await self._append_agent_trace(
                {
                    "step": self._current_fever_step,
                    "agent": self._macnet_decision.name,
                    "role": "MacNet decision - synthesizes graph node outputs",
                    "system_prompt": self._macnet_decision.system_prompt,
                    "input": prompt,
                    "output": output,
                    "tools": ["respond"],
                }
            )
            if obs is not None:
                obs.update(output={"output": output})
            return {"subagent_outputs": {self._macnet_decision.name: output}}

    @staticmethod
    def _format_memory_context_block(context: list[str]) -> str:
        if not context:
            return "# Behavior memory\n(none)"
        return (
            "# Behavior memory\n"
            "Apply only when it matches the current observation. Treat it as a decision rule, not an answer.\n"
            + "\n".join(f"- {item}" for item in context)
        )

    # ==================================================================
    #  DyLAN helpers
    # ==================================================================

    def _init_dylan(self, model: str, config: dict) -> None:
        """Create DyLAN agents (single reusable set), decision agent, ranker, and embedding func."""

        # normalise roles so each row maps to a role
        roles = [r for r in self.dylan_roles if r in DYLAN_VALID_ROLES]
        if not roles:
            roles = ["solver"]

        # agent importance / active state
        self._dylan_importance: dict[str, float] = {}
        self._dylan_active: dict[str, bool] = {}
        self._dylan_nodes: dict[str, AgentNodeBase] = {}

        # ── single reusable agent set (one per role) ──────────────
        self._dylan_agents: list[BaseAgent] = []
        for h in range(self.dylan_height):
            role = roles[h % len(roles)]
            name = f"{role}_{h}"
            agent = BaseAgent(
                name=name,
                system_prompt=DYLAN_ROLE_MAP.get(role, DYLAN_SOLVER_PROMPT),
                llm=GPTChat(model, api_manager=self.api_manager),
            )
            self._dylan_agents.append(agent)
            self._dylan_importance[name] = 1.0 / self.dylan_height
            self._dylan_active[name] = True
            self._dylan_nodes[name] = AgentNodeBase(
                name=name,
                core_agent=agent,
                retriever=EmptyMemoryRetriever(),
                updater=EmptyMemoryUpdater(),
            )

        # ── decision agent ────────────────────────────────────────
        self._dylan_decision = BaseAgent(
            name="dylan_decision",
            system_prompt=DYLAN_DECISION_PROMPT,
            llm=GPTChat(model, api_manager=self.api_manager),
        )

        # ── ranker agent ──────────────────────────────────────────
        self._dylan_ranker = BaseAgent(
            name="dylan_ranker",
            system_prompt=DYLAN_RANKER_PROMPT,
            llm=GPTChat(model, api_manager=self.api_manager),
        )

        # ── embedding function for edge weights ───────────────────
        self._init_embedding(config)

    @staticmethod
    def _dylan_role_from_name(agent_name: str) -> str:
        prefix, _, suffix = agent_name.rpartition("_")
        if prefix and suffix.isdigit():
            return prefix
        return agent_name

    @staticmethod
    def _hotpotqa_prompt_for_dylan_role(role: str) -> str:
        if role == "ground_truth":
            return DYLAN_HOTPOTQA_GROUND_TRUTH_PROMPT
        if role == "decision":
            return DYLAN_HOTPOTQA_DECISION_PROMPT
        return DYLAN_HOTPOTQA_SOLVER_PROMPT

    def _reset_dylan_task_state(self) -> None:
        self._autogen_action_history = []
        if self.graph_name != "dylan":
            return
        self._dylan_active = {name: True for name in self._dylan_active}
        self._dylan_importance = {name: 1.0 / self.dylan_height for name in self._dylan_importance}

    def get_agent_traces(self) -> list[dict[str, Any]]:
        """Return traces for all attempts of the latest task."""
        if self._agent_traces_all_attempts:
            return list(self._agent_traces_all_attempts)
        return list(self._agent_traces)

    def _init_embedding(self, config: dict) -> None:
        """Initialise a lightweight embedding function for cosine similarity."""
        class _EmbedFn:
            def __init__(self, api_manager):
                self._api_manager = api_manager

            def embed(self, texts: list[str]) -> list[list[float]]:
                return self._api_manager.embed_texts_sync(texts)

        self._embed_fn = _EmbedFn(self.api_manager)

    # ==================================================================
    #  DyLAN node factories
    # ==================================================================

    def make_dylan_layer_node(self):
        """Return a reusable LangGraph node function for one DyLAN iteration.

        Reads ``current_layer`` from state; each call is one loop iteration.
        Agents are reused across iterations — only the active_mask controls
        which agents participate.
        """

        async def node(state: DyLANState) -> dict:
            state = dce_copy(state)  # DCE
            task = state["task"].task
            brief = state.get("supervisor_brief", "")
            active_mask: dict[str, bool] = state.get("active_mask", {})
            layer_idx: int = state.get("current_layer", 0)

            # collect agents still active
            col_agents = [
                a for a in self._dylan_agents
                if active_mask.get(a.name, True)
            ]
            if not col_agents:
                return {"consensus_reached": True, "current_layer": layer_idx + 1}

            # build upstream context from previous layer
            prev_outputs: dict[str, str] = {}
            if layer_idx > 0:
                prev_outputs = state.get("layer_outputs", {}).get(layer_idx - 1, {})

            with self._observe(
                f"dylan-layer-{layer_idx}",
                input={
                    "task": task,
                    "layer": layer_idx,
                    "active_agents": [a.name for a in col_agents],
                },
                metadata={"layer_idx": layer_idx, "agent_count": len(col_agents)},
            ) as layer_obs:
                # ── parallel execution with explicit trace context ─
                tctx: dict[str, str] | None = None
                if layer_obs is not None:
                    tctx = {
                        "trace_id": getattr(layer_obs, "trace_id", ""),
                        "parent_span_id": getattr(layer_obs, "id", ""),
                    }
                results = await asyncio.gather(
                    *[
                        self._execute_dylan_agent(agent, task, brief, prev_outputs, tctx)
                        for agent in col_agents
                    ],
                    return_exceptions=True,
                )
                layer_results = {
                    agent.name: ("" if isinstance(result, Exception) else result)
                    for agent, result in zip(col_agents, results)
                }

                # ── edge weights (cosine similarity) ──────────────────
                if layer_idx > 0 and layer_idx < self.dylan_max_layers:
                    self._update_edge_weights(layer_idx, layer_results, state)

                # ── consensus check ───────────────────────────────────
                consensus = self._check_consensus(col_agents, layer_results)

                # ── midpoint ranking / pruning ────────────────────────
                new_active_mask: dict[str, bool] = {}
                if (
                    self.dylan_height > 2
                    and not consensus
                    and layer_idx == (self.dylan_max_layers - 1) // 2
                ):
                    pruned = await self._rank_and_prune(layer_idx, col_agents, layer_results)
                    for a in col_agents:
                        new_active_mask[a.name] = a.name != pruned
                        self._dylan_active[a.name] = a.name != pruned

                if layer_obs is not None:
                    layer_obs.update(
                        output={
                            "consensus": consensus,
                            "agent_count": len(col_agents),
                            "responses": layer_results,
                        }
                    )

            return {
                "layer_outputs": {layer_idx: layer_results},
                "current_layer": layer_idx + 1,
                "consensus_reached": consensus,
                **({"active_mask": new_active_mask} if new_active_mask else {}),
            }

        return node

    async def _execute_dylan_agent(
        self,
        agent: BaseAgent,
        task: str,
        brief: str,
        prev_outputs: dict[str, str],
        trace_context: dict[str, str] | None = None,
    ) -> str:
        node = self._dylan_nodes[agent.name]
        upstream = ""
        if prev_outputs:
            upstream = "# Outputs from other agents in the previous round:\n"
            for name, answer in prev_outputs.items():
                importance = self._dylan_importance.get(name, 0.0)
                upstream += (
                    f"## Agent {name} (importance {importance:.3f}):\n"
                    f"{answer}\n\n"
                )

        # ── trace capture: build full prompt before calling ──
        _captured_prompt_parts: list[str] = []
        def _prompt_builder(memory_context: list[str]) -> str:
            full_prompt = build_dylan_layer_prompt(
                task=task,
                supervisor_brief=brief,
                upstream_context=upstream,
                memory_context=memory_context,
                behavior_memory_context=self._get_memory_context(agent.name),
            )
            _captured_prompt_parts.append(full_prompt)
            return full_prompt

        _, result = await node.run_node(
            task=task,
            prompt_builder=_prompt_builder,
            root_span_name=f"dylan-agent-{agent.name}",
            retrieve_span_name="dylan-agent-memory-retrieve",
            execute_span_name="dylan-agent-core-execution",
            update_span_name="dylan-agent-memory-update",
            trace_context=trace_context,
            root_input={"task": task, "agent": agent.name, "layer_upstream": bool(prev_outputs)},
            root_metadata={"agent": agent.name},
        )
        result = str(result).strip()
        actual_prompt = _captured_prompt_parts[0] if _captured_prompt_parts else ""
        await self._append_agent_trace({
            "step": self._current_fever_step,
            "agent": agent.name,
            "role": f"DyLAN grid agent — {agent.name}",
            "system_prompt": agent.system_prompt,
            "input": actual_prompt,
            "output": result,
            "tools": ["respond"],
        })
        return result

    def _update_edge_weights(
        self,
        layer_idx: int,
        current_outputs: dict[str, str],
        state: DyLANState,
    ) -> None:
        """Compute cosine similarity between current and previous layer outputs."""
        prev_outputs = state.get("layer_outputs", {}).get(layer_idx - 1, {})
        if not prev_outputs:
            return
        try:
            current_names = list(current_outputs.keys())
            current_texts = list(current_outputs.values())
            prev_names = list(prev_outputs.keys())
            prev_texts = list(prev_outputs.values())

            cur_embs = np.array(self._embed_fn.embed(current_texts))
            prev_embs = np.array(self._embed_fn.embed(prev_texts))

            # normalise
            cur_norm = cur_embs / (np.linalg.norm(cur_embs, axis=1, keepdims=True) + 1e-8)
            prev_norm = prev_embs / (np.linalg.norm(prev_embs, axis=1, keepdims=True) + 1e-8)
            sims = cur_norm @ prev_norm.T  # [cur × prev]

            # softmax over prev axis → importance update
            for i, name in enumerate(current_names):
                weights = np.exp(sims[i] - sims[i].max())
                weights /= weights.sum() + 1e-8
                for j, prev_name in enumerate(prev_names):
                    self._dylan_importance[prev_name] += (
                        self._dylan_importance.get(name, 0.0) * float(weights[j]) * 0.01
                    )
        except Exception as exc:
            raise RuntimeError("DyLAN edge embedding update failed") from exc

    def _check_consensus(
        self,
        agents: list[BaseAgent],
        outputs: dict[str, str],
    ) -> bool:
        answers = [_extract_action_line(outputs.get(a.name, "")) for a in agents if outputs.get(a.name)]
        answers = [a for a in answers if a and a != "(no valid action emitted)"]
        if len(answers) < 2:
            return False
        if len(answers) == 2:
            return answers[0] == answers[1]
        counter = Counter(answers)
        max_count = max(counter.values())
        threshold = math.floor(2 / 3 * len(agents))
        return max_count >= threshold

    async def _rank_and_prune(
        self,
        layer_idx: int,
        agents: list[BaseAgent],
        outputs: dict[str, str],
    ) -> str | None:
        """Rank agents by correctness, return the name of the worst one."""
        answers = [(outputs.get(a.name, ""), a) for a in agents if outputs.get(a.name)]
        if len(answers) < 2:
            return None

        responses_text = ""
        for i, (ans, _) in enumerate(answers):
            responses_text += f"Agent{i+1}: {ans}\n"
        prompt = build_dylan_ranker_prompt(responses_text)

        rank_str = await self._dylan_ranker.respond(prompt)
        # parse "1 > 3 > 2" style
        import re
        matches = re.findall(r"Agent\s*(\d+)", rank_str)
        if not matches:
            matches = re.findall(r"\b(\d+)\b", rank_str)
        indices = [int(m) - 1 for m in matches]

        if len(indices) != len(answers):
            return None

        loser_idx = indices[-1]
        if 0 <= loser_idx < len(answers):
            loser_name = answers[loser_idx][1].name
            self._dylan_active[loser_name] = False
            self._dylan_importance[loser_name] = 0.0
            return loser_name
        return None

    def dylan_router(self, state: DyLANState) -> str:
        """Conditional edge router: consensus or max layers → final_decision, else → loop back."""
        if state.get("consensus_reached"):
            return "final_decision"
        if state.get("current_layer", 0) >= self.dylan_max_layers:
            return "final_decision"
        return "next_layer"

    async def dylan_decision_node(self, state: DyLANState) -> dict:
        """Summarise the last DyLAN layer's outputs into a final action."""
        state = dce_copy(state)  # DCE
        task = state["task"].task
        layer_outputs = state.get("layer_outputs", {})
        if not layer_outputs:
            return {"final_output": "No agent responses."}

        last_layer = max(layer_outputs.keys())
        outputs = layer_outputs[last_layer]

        # if only one active answer, return it directly
        if len(outputs) == 1:
            result = str(list(outputs.values())[0]).strip()
            await self._append_agent_trace({
                "step": self._current_fever_step,
                "agent": self._dylan_decision.name,
                "role": "Decision Agent — synthesises last DyLAN layer outputs into a single action",
                "system_prompt": self._dylan_decision.system_prompt,
                "input": f"(single active agent, bypassed LLM) layer={last_layer}",
                "output": result,
                "tools": ["respond"],
            })
            return {"final_output": result}

        responses_text = ""
        for i, (name, answer) in enumerate(outputs.items()):
            responses_text += f"Agent{i+1} ({name}): {answer}\n"

        prompt = build_dylan_decision_prompt(
            task,
            state.get("supervisor_brief", ""),
            responses_text,
            self._get_memory_context(self._dylan_decision.name),
        )
        with self._observe(
            "dylan-decision",
            input={"task": task, "layer": last_layer},
            metadata={"agent": self._dylan_decision.name},
        ) as obs:
            result = str(await self._dylan_decision.respond(prompt)).strip()
            await self._append_agent_trace({
                "step": self._current_fever_step,
                "agent": self._dylan_decision.name,
                "role": "Decision Agent — synthesises last DyLAN layer outputs into a single action",
                "system_prompt": self._dylan_decision.system_prompt,
                "input": prompt,
                "output": result,
                "tools": ["respond"],
            })
            if obs is not None:
                obs.update(output={"decision": result, "action": _extract_action_line(result), "reason": _extract_reason_line(result)})
            return {"final_output": result}

    async def dylan_final_node(self, state: DyLANState) -> dict:
        """Post-process the DyLAN decision into the final structured output."""
        state = dce_copy(state)  # DCE
        final_output = state.get("final_output", "")
        task = state["task"].task
        brief = state.get("supervisor_brief", "")
        if self._is_fever_task(state["task"]):
            prompt = build_dylan_final_fever_prompt(task, brief, final_output, self._get_memory_context(self.final_agent.name))
        elif self._is_hotpotqa_task(state["task"]):
            prompt = build_dylan_final_hotpotqa_prompt(task, brief, final_output, self._get_memory_context(self.final_agent.name))
        else:
            prompt = build_dylan_final_prompt(task, brief, final_output, self._get_memory_context(self.final_agent.name))
        with self._observe(
            "dylan-final",
            input={"task": task},
            metadata={"agent": self.final_agent.name},
        ) as obs:
            result = str(await self.final_agent.respond(prompt)).strip()
            await self._append_agent_trace({
                "step": self._current_fever_step,
                "agent": self.final_agent.name,
                "role": "Final Agent — formats the DyLAN decision into the final answer",
                "system_prompt": self.final_agent.system_prompt,
                "input": prompt,
                "output": result,
                "tools": ["respond"],
            })
            if obs is not None:
                obs.update(output={"final_output": result, "action": _extract_action_line(result), "reason": _extract_reason_line(result)})
            return {"final_output": result}

    # ==================================================================
    #  Run (multi‑round + trace recording)
    # ==================================================================

    async def run(
        self,
        task: str,
        metadata: dict | None = None,
        max_rounds: int = 1,
        max_steps: int = 10,
    ) -> tuple[MASState | DyLANState, TraceRecorder]:
        _edbg(f"[ENGINE:run] task={task[:80]} memory={self.mas_memory_backend}")
        final_result: MASState | DyLANState | dict[str, Any] = {}
        final_task_input: TaskInput | None = None
        base_task_metadata = dict(metadata or {})
        self._begin_task_api_scope(base_task_metadata)
        if self._memory_store is not None:
            memory_task_metadata = dict(base_task_metadata)
            memory_task_metadata["graph_name"] = self.graph_name
            await self._memory_store.begin_task(
                TaskInput(task=task, metadata=memory_task_metadata),
                agent_ids=self._memory_agent_ids(),
                api_manager=self.api_manager,
            )
        recorder = TraceRecorder(
            graph_name=self.graph_name,
            model_name=self.model_name or self.api_manager.model_name,
            max_rounds=max_rounds,
        )

        for round_idx in range(max_rounds):
            task_metadata = dict(base_task_metadata)
            task_metadata["graph_name"] = self.graph_name
            task_input = TaskInput(task=task, metadata=task_metadata)
            final_task_input = task_input
            result = await self._execute_task_attempt(
                task_input=task_input,
                max_steps=max_steps,
                recorder=recorder,
                attempt_idx=round_idx,
            )
            final_result = result

        if (
            self._memory_store is not None
            and not self.read_only
            and final_task_input is not None
        ):
            await self._memory_store.record_task(
                final_task_input,
                dict(final_result),
                agent_traces=self.get_agent_traces(),
                success=self._is_task_success(final_task_input, dict(final_result)),
                api_manager=self.api_manager,
            )

        self._reconcile_token_usage(recorder)
        return final_result, recorder

    async def _run_general_task(self, task_input: TaskInput) -> MASState | DyLANState:
        if self._memory_store is not None and not self._memory_agent_contexts:
            self._memory_agent_contexts = await self._retrieve_memory_contexts(task_input)
        if self.graph_name == "dylan":
            initial_state: DyLANState = {
                "task": task_input,
                "layer_outputs": {},
                "active_mask": {name: True for name in self._dylan_active},
                "current_layer": 0,
            }
        else:
            initial_state = {
                "task": task_input,
                "subagent_outputs": {},
                "subagent_memory": {},
            }

        with self._observe(
            self.graph_name,
            input={"task": task_input.task, "graph": self.graph_name},
            metadata={"graph": self.graph_name, "dataset": task_input.metadata.get("dataset")},
        ) as graph_observation:
            with self._push_trace_parent(graph_observation):
                return await self.graph.ainvoke(initial_state)  # type: ignore[arg-type]

    async def _run_fever_episode(self, task_input: TaskInput, max_steps: int) -> dict:
        from src.datasets.fever import build_fever_step_prompt  # lazy — fever only
        from src.eval.fever import FeverEnv                      # lazy — fever only
        _edbg(f"[ENGINE:fever_episode] start claim={task_input.task[:80]}")
        self._current_fever_step = 0
        env = FeverEnv(answer=str(task_input.metadata.get("label", "")), max_steps=max_steps)
        trajectory: list[dict[str, str]] = []
        last_result: dict = {}
        pending_finish_answer: str | None = None

        original_final_prompt = self.final_agent.system_prompt
        self.final_agent.system_prompt = FEVER_FINAL_PROMPT
        original_solver_prompts = {}
        if self.graph_name == "dylan":
            for agent in self._dylan_agents:
                original_solver_prompts[agent.name] = agent.system_prompt
                agent.system_prompt = DYLAN_FEVER_SOLVER_PROMPT
        try:
            for step_idx in range(max_steps):
                self._current_fever_step = step_idx
                with self._observe(
                    "fever-step",
                    input={"claim": task_input.task, "step": step_idx + 1},
                    metadata={"dataset": "fever", "step": step_idx + 1},
                ) as step_obs:
                    with self._push_trace_parent(step_obs):
                        step_prompt = build_fever_step_prompt(task_input.task, trajectory)
                        graph_metadata = dict(task_input.metadata)
                        recent = (
                            "\n".join(
                                f"Action {idx + 1}: {turn.get('action', '')}\nObservation {idx + 1}: {turn.get('observation', '')}"
                                for idx, turn in enumerate(trajectory)
                            )
                            if trajectory
                            else "(none yet)"
                        )
                        graph_metadata["supervisor_task"] = (
                            f"Claim: {task_input.task}\n"
                            f"Step: {step_idx + 1}\n"
                            f"Recent trajectory:\n{recent}"
                        )
                        graph_task = TaskInput(task=step_prompt, metadata=graph_metadata)
                        if self._memory_store is not None:
                            self._memory_agent_contexts = await self._retrieve_memory_contexts(graph_task)
                        last_result = await self._run_general_task(graph_task)
                        raw_output = last_result.get("final_output", "")
                        action = FeverEnv.process_action(raw_output)
                        action_type, action_argument = FeverEnv.parse_action(action)
                        if action_type == "Finish":
                            trajectory.append(
                                {
                                    "action": action,
                                    "observation": "(evaluation withheld)",
                                    "raw_output": raw_output,
                                }
                            )
                            pending_finish_answer = str(action_argument or "").strip()
                            observation, reward, done = "(evaluation withheld)", 0.0, True
                        else:
                            observation, reward, done = env.step(raw_output)
                            trajectory.append(
                                {
                                    "action": action,
                                    "observation": observation,
                                    "raw_output": raw_output,
                                }
                            )
                        if step_obs is not None:
                            step_obs.update(
                                output={
                                    "action": action,
                                    "observation": observation,
                                    "done": done,
                                }
                            )
                        if done:
                            break
        finally:
            self.final_agent.system_prompt = original_final_prompt
            if self.graph_name == "dylan":
                for agent in self._dylan_agents:
                    agent.system_prompt = original_solver_prompts.get(
                        agent.name,
                        DYLAN_ROLE_MAP.get(agent.name.split("_")[0], DYLAN_SOLVER_PROMPT),
                    )

        return {
            **last_result,
            "final_output": pending_finish_answer or TIMEOUT_SENTINEL,
            "predicted_label": pending_finish_answer or TIMEOUT_SENTINEL,
            "reward": 0.0,
            "done": False,
            "terminated": pending_finish_answer is not None,
            "feedback": "(evaluation pending)",
            "steps_taken": len(trajectory),
            "trajectory": trajectory,
            "_pending_finish_answer": pending_finish_answer,
        }

    async def _run_hotpotqa_episode(
        self,
        task_input: TaskInput,
        max_steps: int,
    ) -> dict:
        from src.datasets.hotpotqa import build_hotpotqa_step_prompt  # lazy — hotpotqa only
        from src.eval.hotpotqa_env import HotpotQAEnv                 # lazy — hotpotqa only

        question = str(task_input.metadata.get("question") or task_input.task)
        answer = str(task_input.metadata.get("answer", ""))
        _edbg(f"[ENGINE:hotpotqa_episode] start question={question[:80]}")
        self._current_fever_step = 0
        env = HotpotQAEnv(answer=answer, max_steps=max_steps)
        trajectory: list[dict[str, str]] = []
        last_result: dict = {}
        pending_finish_answer: str | None = None

        original_final_prompt = self.final_agent.system_prompt
        self.final_agent.system_prompt = HOTPOTQA_FINAL_PROMPT
        original_dylan_decision_prompt = None
        if self.graph_name == "dylan":
            original_dylan_decision_prompt = self._dylan_decision.system_prompt
            self._dylan_decision.system_prompt = DYLAN_HOTPOTQA_AGGREGATOR_PROMPT
        original_solver_prompts = {}
        if self.graph_name == "dylan":
            for agent in self._dylan_agents:
                original_solver_prompts[agent.name] = agent.system_prompt
                role = self._dylan_role_from_name(agent.name)
                agent.system_prompt = self._hotpotqa_prompt_for_dylan_role(role)
        try:
            for step_idx in range(len(trajectory), max_steps):
                self._current_fever_step = step_idx
                with self._observe(
                    "hotpotqa-step",
                    input={"question": question, "step": step_idx + 1},
                    metadata={"dataset": "hotpotqa", "step": step_idx + 1},
                ) as step_obs:
                    with self._push_trace_parent(step_obs):
                        step_prompt = build_hotpotqa_step_prompt(question, trajectory)
                        graph_metadata = dict(task_input.metadata)
                        recent_actions = (
                            "\n".join(
                                f"{idx + 1}. {turn.get('action', '')}"
                                for idx, turn in enumerate(trajectory)
                            )
                            if trajectory
                            else "(none yet)"
                        )
                        recent_feedback = (
                            "\n".join(
                                f"Action {idx + 1}: {turn.get('action', '')}\nObservation {idx + 1}: {turn.get('observation', '')}"
                                for idx, turn in enumerate(trajectory)
                            )
                            if trajectory
                            else "(none yet)"
                        )
                        graph_metadata["supervisor_task"] = (
                            f"Question: {question}\n"
                            f"Step: {step_idx + 1}\n"
                            f"Committed action history:\n{recent_actions}\n\n"
                            f"Environment feedback from recent actions:\n{recent_feedback}"
                        )
                        graph_task = TaskInput(task=step_prompt, metadata=graph_metadata)
                        if self._memory_store is not None:
                            self._memory_agent_contexts = await self._retrieve_memory_contexts(graph_task)
                        last_result = await self._run_general_task(graph_task)
                        raw_output = last_result.get("final_output", "")
                        action = _extract_search_lookup_finish_action(raw_output)
                        action_type, action_argument = HotpotQAEnv.parse_action(action)
                        if _is_placeholder_hotpotqa_action(action):
                            observation = (
                                "Invalid placeholder action. Replace bracket content with a concrete "
                                "query, exact visible title, or supported answer span."
                            )
                            reward, done = 0.0, False
                            trajectory.append(
                                {
                                    "action": action,
                                    "observation": observation,
                                    "raw_output": raw_output,
                                }
                            )
                        elif action_type == "Finish":
                            trajectory.append(
                                {
                                    "action": action,
                                    "observation": "(evaluation withheld)",
                                    "raw_output": raw_output,
                                }
                            )
                            pending_finish_answer = str(action_argument or "").strip()
                            observation, reward, done = "(evaluation withheld)", 0.0, True
                        else:
                            observation, reward, done = env.step(raw_output)
                            trajectory.append(
                                {
                                    "action": action,
                                    "observation": observation,
                                    "raw_output": raw_output,
                                }
                            )
                        if step_obs is not None:
                            step_obs.update(
                                output={
                                    "action": action,
                                    "observation": observation,
                                    "done": done,
                                }
                            )
                        if done:
                            break
        finally:
            self.final_agent.system_prompt = original_final_prompt
            if self.graph_name == "dylan":
                self._dylan_decision.system_prompt = (
                    original_dylan_decision_prompt or DYLAN_DECISION_PROMPT
                )
                for agent in self._dylan_agents:
                    agent.system_prompt = original_solver_prompts.get(
                        agent.name,
                        DYLAN_ROLE_MAP.get(
                            self._dylan_role_from_name(agent.name),
                            DYLAN_SOLVER_PROMPT,
                        ),
                    )

        return {
            **last_result,
            "final_output": pending_finish_answer or TIMEOUT_SENTINEL,
            "predicted_answer": pending_finish_answer or TIMEOUT_SENTINEL,
            "reward": 0.0,
            "final_f1": 0.0,
            "done": False,
            "terminated": pending_finish_answer is not None,
            "feedback": "(evaluation pending)",
            "steps_taken": len(trajectory),
            "trajectory": trajectory,
            "_pending_finish_answer": pending_finish_answer,
        }

    async def supervisor_node(self, state: MASState) -> dict:
        state = dce_copy(state)  # DCE
        task_input = state["task"]
        supervisor_task = str(task_input.metadata.get("supervisor_task") or task_input.task)
        prompt = build_supervisor_prompt(supervisor_task, self._get_memory_context(self.supervisor.name))
        with self._observe("supervisor-node", input={"task": supervisor_task, "prompt": prompt}, metadata={"agent": self.supervisor.name}) as obs:
            result = await self.supervisor.respond(prompt)
            await self._append_agent_trace({
                "step": self._current_fever_step,
                "agent": self.supervisor.name,
                "role": "Supervisor — decomposes the task into sub-tasks and briefs sub-agents",
                "system_prompt": self.supervisor.system_prompt,
                "input": prompt,
                "output": result,
                "tools": ["respond"],
            })
            if obs is not None:
                obs.update(output={"supervisor_brief": result})
            return {"supervisor_brief": result}

    def make_subagent_node(self, subagent_name: str):
        async def node(state: MASState) -> dict:
            state = dce_copy(state)  # DCE
            task = state["task"].task
            supervisor_brief = state.get("supervisor_brief", "")
            memory_context, output = await self.subagents[subagent_name].run(
                task,
                supervisor_brief,
                external_memory_context=self._get_memory_context(subagent_name),
            )
            return {
                "subagent_outputs": {subagent_name: output},
                "subagent_memory": {subagent_name: memory_context},
            }
        return node

    async def final_node(self, state: MASState) -> dict:
        task = state["task"].task
        brief = state.get("supervisor_brief", "")
        outputs = state.get("subagent_outputs", {})
        rendered_outputs = "\n".join(f"{name}: {value}" for name, value in outputs.items())
        if self._is_fever_task(state["task"]):
            prompt = build_final_fever_prompt(task, brief, rendered_outputs, self._get_memory_context(self.final_agent.name))
        elif self._is_hotpotqa_task(state["task"]):
            prompt = build_final_hotpotqa_prompt(task, brief, rendered_outputs, self._get_memory_context(self.final_agent.name))
        else:
            prompt = build_final_prompt(task, brief, rendered_outputs, self._get_memory_context(self.final_agent.name))
        with self._observe("final-node", input={"task": task, "prompt": prompt}, metadata={"agent": self.final_agent.name}) as obs:
            result = await self.final_agent.respond(prompt)
            await self._append_agent_trace({
                "step": self._current_fever_step,
                "agent": self.final_agent.name,
                "role": "Final Agent - formats upstream MAS output into the committed environment action",
                "system_prompt": self.final_agent.system_prompt,
                "input": prompt,
                "output": result,
                "tools": ["respond"],
            })
            if obs is not None:
                obs.update(output={"final_output": result, "action": _extract_action_line(result), "reason": _extract_reason_line(result)})
            return {"final_output": result}
