import argparse
import asyncio
import json as _json
import os
from datetime import datetime, timezone
from pathlib import Path as _Path

from src.core.types import TaskInput
from src.datasets import load_dataset_tasks
from src.eval import (
    FeverDatasetEvaluator,
    HotpotQADatasetEvaluator,
    evaluate_task_result,
)
from src.api import flush_langfuse, init_langfuse
from src.configs import load_config
from src.log import logger, setup
from src.runtime import MinimalLangGraphRuntime
from src.memory.native_mas_memory import (
    NATIVE_MAS_MEMORY_CHOICES,
    NATIVE_MAS_MEMORY_VERSION,
    normalize_mas_memory_backend,
)


def _dbg(msg: str) -> None:
    logger.debug(msg)


# ---------------------------------------------------------------------------
#  result JSONL writer  (async-safe)
# ---------------------------------------------------------------------------

_result_lock = asyncio.Lock()
_result_path: str | None = None


def _resolve_cli_api_profile(args: argparse.Namespace) -> str | None:
    api_profile = getattr(args, "api_profile", None)
    if api_profile:
        return api_profile
    profiles = load_config().get("api_profiles", {}) or {}
    if args.model in profiles:
        return args.model
    return os.getenv("OPENAI_API_PROFILE_ID") or os.getenv("OPENAI_API_PROFILE")


def _init_result_file(
    explicit_path: str | None = None,
    *,
    mas_memory_backend: str = "none",
) -> str:
    """Create the method-specific result directory and return a JSONL path."""
    if explicit_path:
        file_path = _Path(explicit_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            file_path.unlink()
        return str(file_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_group = mas_memory_backend if mas_memory_backend != "none" else "non-memory"
    dir_path = _Path("result") / result_group
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"run_{ts}.jsonl"
    return str(file_path)


async def _write_header(path: str, args: argparse.Namespace, *, task_count: int) -> None:
    """Write the header (config / params) as the first JSONL line."""
    config = {
        "type": "config",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            "dataset": args.dataset,
            "graph": args.graph,
            "model": args.model or os.getenv("OPENAI_MODEL_NAME") or "(api profile/config default)",
            "api_profile_id": _resolve_cli_api_profile(args),
            "num_tasks": task_count,
            "task_indices": args.task_indices,
            "dylan_max_layers": args.dylan_max_layers,
            "dylan_height": args.dylan_height,
            "subagents": args.subagents,
            "rounds": args.rounds,
            "max_steps": args.max_steps,
            "mode": args.mode,
            "concurrency": args.concurrency,
            "mas_memory_backend": getattr(args, "mas_memory_backend", "none"),
            "mas_memory_implementation": (
                NATIVE_MAS_MEMORY_VERSION
                if getattr(args, "mas_memory_backend", "none") != "none"
                else None
            ),
            "mas_memory_path": getattr(args, "mas_memory_source", None),
            "mas_memory_topk": getattr(args, "mas_memory_topk", 0),
            "result_path": getattr(args, "result_path", None),
        },
    }
    async with _result_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(config, ensure_ascii=False) + "\n")
    _dbg(f"[MAIN] header written to {path}")


async def _write_task_result(
    path: str,
    task: TaskInput,
    result: dict,
    recorder,
    evaluation: dict,
    agent_traces: list[dict],
) -> None:
    """Write one task result (with full agent I/O) as a JSONL line."""
    traces_out = []
    for t in agent_traces:
        traces_out.append({
            "attempt": t.get("attempt", 0),
            "step": t.get("step", 0),
            "agent": t.get("agent", "?"),
            "role": t.get("role", "?"),
            "system_prompt": str(t.get("system_prompt", "")),
            "input": str(t.get("input", "")),
            "output": str(t.get("output", "")),
        })

    rounds_out = []
    token_events_out = []
    first_success_round = None
    initial_round_success = False
    for r in recorder.rounds:
        round_payload = {
            "round_idx": r.round_idx,
            "final_output": r.final_output,
            "predicted_answer": str(getattr(r, "predicted_answer", "")),
            "success": bool(r.success),
            "done": bool(getattr(r, "done", False)),
            "terminated": bool(getattr(r, "terminated", False)),
            "feedback": str(getattr(r, "feedback", "")),
            "steps_taken": getattr(r, "steps_taken", None),
            "trajectory": [
                dict(turn)
                for turn in (getattr(r, "trajectory", []) or [])
                if isinstance(turn, dict)
            ],
            "prompt_tokens": int(r.prompt_tokens),
            "completion_tokens": int(r.completion_tokens),
            "duration_ms": float(r.duration_ms),
            "metadata": dict(r.metadata or {}),
        }
        rounds_out.append(round_payload)
        if r.round_idx == 0:
            initial_round_success = bool(r.success)
        if first_success_round is None and r.success:
            first_success_round = r.round_idx
    for event in getattr(recorder, "token_events", []) or []:
        token_events_out.append(
            {
                "name": str(event.name),
                "prompt_tokens": int(event.prompt_tokens),
                "completion_tokens": int(event.completion_tokens),
                "duration_ms": float(event.duration_ms),
                "metadata": dict(event.metadata or {}),
            }
        )

    record = {
        "type": "task",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_index": task.metadata.get("task_index", "?"),
        "question": task.task,
        "gold_answer": (
            evaluation.get("gold_answer")
            or evaluation.get("gold_label")
            or task.metadata.get("answer")
            or task.metadata.get("label", "?")
        ),
        "predicted_answer": (
            evaluation.get("predicted_answer")
            or evaluation.get("predicted_label")
            or result.get("final_output", "")
        ),
        "exact_match": evaluation.get("exact_match", False),
        "f1": evaluation.get("f1", 0.0),
        "done": evaluation.get("done", result.get("done", False)),
        "reward": evaluation.get("reward", result.get("reward", 0.0)),
        "progress": evaluation.get("progress", result.get("progress")),
        "finished_subgoals": evaluation.get("finished_subgoals", result.get("finished_subgoals", [])),
        "unfinished_subgoals": evaluation.get("unfinished_subgoals", result.get("unfinished_subgoals", [])),
        "question_type": task.metadata.get("type", "?"),
        "level": task.metadata.get("level", "?"),
        "supervisor_brief": result.get("supervisor_brief", ""),
        "final_output": result.get("final_output", ""),
        "prompt_tokens": recorder.total_prompt_tokens,
        "completion_tokens": recorder.total_completion_tokens,
        "duration_ms": int(recorder.total_duration_ms),
        "round_count": len(rounds_out),
        "rounds": rounds_out,
        "token_events": token_events_out,
        "initial_round_success": initial_round_success,
        "first_success_round": first_success_round,
        "agent_traces": traces_out,
    }
    async with _result_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(record, ensure_ascii=False) + "\n")
    _dbg(f"[MAIN] task {task.metadata.get('task_index','?')} written to {path}")


async def _write_summary(
    path: str,
    fever_evaluator,
    hotpotqa_evaluator,
) -> None:
    """Write the final summary line to the JSONL."""
    summary = {"type": "summary", "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    if fever_evaluator is not None:
        summary["fever_summary"] = fever_evaluator.summary()
    if hotpotqa_evaluator is not None:
        summary["hotpotqa_summary"] = hotpotqa_evaluator.summary()
    async with _result_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(summary, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------


def _select_tasks_by_indices(tasks: list[TaskInput], indices: list[int]) -> list[TaskInput]:
    if not indices:
        return tasks
    selected: list[TaskInput] = []
    for idx in indices:
        if idx < 0 or idx >= len(tasks):
            raise IndexError(f"task index {idx} out of range for loaded task set of size {len(tasks)}")
        selected.append(tasks[idx])
    return selected


async def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal LangGraph MAS scaffold.")
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "fever"],
        default="hotpotqa",
    )
    parser.add_argument("--graph", type=str, default="dylan")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-profile", type=str, default=None, help="API profile id from configs.yaml, e.g. gpt-4o-mini")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--subagents", type=int, default=4)
    parser.add_argument("--dylan-max-layers", type=int, default=5)
    parser.add_argument("--dylan-height", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--task-indices", type=int, nargs="*", default=None, help="Explicit dataset task indices to run, e.g. --task-indices 1 7 12")
    parser.add_argument(
        "--mas-memory-backend",
        type=str,
        default="none",
        choices=NATIVE_MAS_MEMORY_CHOICES,
        help="Native online MAS memory backend for isolated experiments.",
    )
    parser.add_argument(
        "--mas-memory-path",
        "--mas-memory-source",
        dest="mas_memory_source",
        type=str,
        default=None,
        help="Writable state directory for the selected native MAS memory backend.",
    )
    parser.add_argument("--mas-memory-topk", type=int, default=2, help="Retrieved native memory snippets per agent.")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="train updates memory; test only runs retrieval/injection without memory writes")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of tasks to run concurrently (default 1 = sequential)")
    parser.add_argument(
        "--result-path",
        type=str,
        default=None,
        help="Explicit JSONL result path. Useful for parallel experiment runners.",
    )
    args = parser.parse_args()
    args.mas_memory_backend = normalize_mas_memory_backend(args.mas_memory_backend)

    setup(args.graph)
    logger.info("Langfuse enabled: {}", init_langfuse())
    if args.task and args.task_indices:
        raise ValueError("--task and --task-indices cannot be used together")
    if args.mas_memory_backend != "none" and args.concurrency > 1:
        raise ValueError(
            "Native MAS memory backends require --concurrency 1 so each task observes "
            "all prior writes in a deterministic order."
        )

    if args.task:
        tasks = [TaskInput(task=args.task, metadata={})]
    else:
        if args.task_indices:
            load_limit = max(args.task_indices) + 1
            tasks = load_dataset_tasks(args.dataset, limit=load_limit)
            tasks = _select_tasks_by_indices(tasks, args.task_indices)
        else:
            tasks = load_dataset_tasks(args.dataset, limit=args.num_tasks)

    fever_evaluator = FeverDatasetEvaluator() if args.dataset == "fever" and not args.task else None
    hotpotqa_evaluator = HotpotQADatasetEvaluator() if args.dataset == "hotpotqa" and not args.task else None
    if args.graph == "dylan":
        logger.info("DyLAN loop: max {} iterations x {} agents", args.dylan_max_layers, args.dylan_height)

    _dbg(f"[MAIN] concurrency={args.concurrency} tasks={len(tasks)} memory={args.mas_memory_backend}")

    # ── init result file ──────────────────────────────────────────
    result_path = _init_result_file(
        args.result_path,
        mas_memory_backend=args.mas_memory_backend,
    )
    await _write_header(result_path, args, task_count=len(tasks))

    if args.concurrency > 1 and len(tasks) > 1:
        _dbg(f"[MAIN] entering concurrent path")
        sem = asyncio.Semaphore(args.concurrency)

        async def _run_concurrent(selected_task: TaskInput):
            async with sem:
                _dbg(f"[CONCURRENT] start task idx={selected_task.metadata.get('task_index','?')}")
                rt = MinimalLangGraphRuntime(
                    graph_name=args.graph,
                    subagent_count=args.subagents,
                    model_name=args.model,
                    api_profile_id=args.api_profile,
                    dylan_max_layers=args.dylan_max_layers,
                    dylan_height=args.dylan_height,
                    read_only=args.mode == "test",
                    mas_memory_backend=args.mas_memory_backend,
                    mas_memory_source=args.mas_memory_source,
                    mas_memory_topk=args.mas_memory_topk,
                )
                result, recorder = await rt.run(
                    selected_task.task,
                    metadata=selected_task.metadata,
                    max_rounds=args.rounds,
                    max_steps=args.max_steps,
                )
                evaluation = evaluate_task_result(selected_task, result)
                traces = rt.get_agent_traces()
                await _write_task_result(result_path, selected_task, result, recorder, evaluation, traces)
                return selected_task, result, recorder, evaluation

        _dbg(f"[MAIN] launching asyncio.gather for {len(tasks)} tasks")
        all_results = await asyncio.gather(*[_run_concurrent(t) for t in tasks])
        _dbg(f"[MAIN] asyncio.gather returned {len(all_results)} results")
        for selected_task, result, recorder, evaluation in all_results:
            logger.info("=== Task ===\n{}", selected_task.metadata.get("task_index", selected_task.task[:120]))
            logger.info("=== Final Output ===\n{}", result.get("final_output", ""))
            logger.info("=== Eval ===\n{}", evaluation)
            logger.info("=== Trace Stats ===\n{}", recorder.format_table())
            if fever_evaluator is not None:
                fever_evaluator.add(selected_task, result, prompt_tokens=recorder.total_prompt_tokens, completion_tokens=recorder.total_completion_tokens, duration_ms=recorder.total_duration_ms)
            if hotpotqa_evaluator is not None:
                hotpotqa_evaluator.add(selected_task, result, prompt_tokens=recorder.total_prompt_tokens, completion_tokens=recorder.total_completion_tokens, duration_ms=recorder.total_duration_ms)
    else:
        _dbg(f"[MAIN] entering serial path ({len(tasks)} tasks)")
        runtime = MinimalLangGraphRuntime(
            graph_name=args.graph,
            subagent_count=args.subagents,
            model_name=args.model,
            api_profile_id=args.api_profile,
            dylan_max_layers=args.dylan_max_layers,
            dylan_height=args.dylan_height,
            read_only=args.mode == "test",
            mas_memory_backend=args.mas_memory_backend,
            mas_memory_source=args.mas_memory_source,
            mas_memory_topk=args.mas_memory_topk,
        )
        for selected_task in tasks:
            _dbg(f"[SERIAL] start task idx={selected_task.metadata.get('task_index','?')}")
            result, recorder = await runtime.run(
                selected_task.task,
                metadata=selected_task.metadata,
                max_rounds=args.rounds,
                max_steps=args.max_steps,
            )
            evaluation = evaluate_task_result(selected_task, result)
            traces = runtime.get_agent_traces()
            # ── write immediately after each task ──
            await _write_task_result(result_path, selected_task, result, recorder, evaluation, traces)

            logger.info("=== Task ===\n{}", selected_task.metadata.get("task_index", selected_task.task[:120]))
            logger.info("=== Final Output ===\n{}", result.get("final_output", ""))
            logger.info("=== Eval ===\n{}", evaluation)
            logger.info("=== Trace Stats ===\n{}", recorder.format_table())
            _dbg(f"[SERIAL] done task idx={selected_task.metadata.get('task_index','?')}")
            if fever_evaluator is not None:
                fever_evaluator.add(selected_task, result, prompt_tokens=recorder.total_prompt_tokens, completion_tokens=recorder.total_completion_tokens, duration_ms=recorder.total_duration_ms)
            if hotpotqa_evaluator is not None:
                hotpotqa_evaluator.add(selected_task, result, prompt_tokens=recorder.total_prompt_tokens, completion_tokens=recorder.total_completion_tokens, duration_ms=recorder.total_duration_ms)

    # ── final summary ─────────────────────────────────────────────
    await _write_summary(result_path, fever_evaluator, hotpotqa_evaluator)
    _dbg(f"[MAIN] all tasks done, flush+summary")
    if fever_evaluator is not None:
        logger.info("=== FEVER Summary ===\n{}", fever_evaluator.summary())
    if hotpotqa_evaluator is not None:
        logger.info("=== HotpotQA Summary ===\n{}", hotpotqa_evaluator.summary())
    flush_langfuse()
    _dbg(f"[MAIN] exit")


if __name__ == "__main__":
    asyncio.run(main())
