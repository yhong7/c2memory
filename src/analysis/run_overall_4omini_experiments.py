from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.hotpotqa_env import compute_em as hotpot_em
from src.eval.hotpotqa_env import compute_f1 as hotpot_f1
from src.memory.native_mas_memory import NATIVE_MAS_MEMORY_VERSION


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "result" / "final" / "c2mem-result-4o-mini"

GRAPHS = ["llm_debate", "macnet", "autogen"]
DATASETS = {
    "hotpotqa": 50,
    "fever": 50,
}
MEMORIES = ["c2mem"]
MEMORY_CHOICES = list(MEMORIES)
EXTERNAL_BACKEND = {"c2mem": "c2mem"}


@dataclass(frozen=True)
class Experiment:
    dataset: str
    graph: str
    memory: str
    num_tasks: int

    @property
    def combo_id(self) -> str:
        return f"{self.dataset}__{self.graph}__{self.memory}"


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_runner_event(out_root: Path, payload: dict[str, Any]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "runner_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _round_prediction(record: dict[str, Any], round_index: int = 0) -> str:
    rounds = record.get("rounds")
    if isinstance(rounds, list) and len(rounds) > round_index and isinstance(rounds[round_index], dict):
        round_payload = rounds[round_index]
        return str(round_payload.get("predicted_answer") or round_payload.get("final_output") or "")
    return str(record.get("predicted_answer") or record.get("final_output") or "")


def _score_prediction(dataset: str, prediction: str, gold: str) -> tuple[bool, float]:
    if dataset == "fever":
        pred = str(prediction).strip().upper()
        target = str(gold).strip().upper()
        exact = bool(pred) and pred == target
        return exact, 1.0 if exact else 0.0
    exact = bool(hotpot_em(prediction, gold))
    return exact, float(hotpot_f1(prediction, gold))


def _final_success(dataset: str, record: dict[str, Any]) -> bool:
    if dataset == "fever":
        return float(record.get("reward", 0.0) or 0.0) >= 1.0
    return bool(record.get("exact_match"))


def _is_max_step_failure(record: dict[str, Any], *, dataset: str, max_steps: int) -> bool:
    if _final_success(dataset, record):
        return False
    rounds = record.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return False
    last = rounds[-1] if isinstance(rounds[-1], dict) else {}
    if bool(last.get("success")):
        return False
    try:
        steps_taken = int(last.get("steps_taken") or 0)
    except (TypeError, ValueError):
        steps_taken = 0
    return steps_taken >= max_steps and not bool(last.get("done"))


def summarize_result(path: Path, exp: Experiment, *, max_steps: int) -> dict[str, Any]:
    rows = _jsonl_rows(path)
    tasks = [row for row in rows if row.get("type") == "task"]
    initial_em_values: list[float] = []
    initial_f1_values: list[float] = []
    final_em_values: list[float] = []
    final_f1_values: list[float] = []
    max_step_failures = 0
    prompt_tokens = 0
    completion_tokens = 0

    for record in tasks:
        gold = str(record.get("gold_answer") or "")
        pred0 = _round_prediction(record, 0)
        initial_em, initial_f1 = _score_prediction(exp.dataset, pred0, gold)
        initial_em_values.append(1.0 if initial_em else 0.0)
        initial_f1_values.append(initial_f1)
        final_success = _final_success(exp.dataset, record)
        final_em_values.append(1.0 if final_success else 0.0)
        if exp.dataset == "fever":
            final_f1_values.append(1.0 if final_success else 0.0)
        else:
            final_f1_values.append(float(record.get("f1", 0.0) or 0.0))
        prompt_tokens += int(record.get("prompt_tokens", 0) or 0)
        completion_tokens += int(record.get("completion_tokens", 0) or 0)
        if _is_max_step_failure(record, dataset=exp.dataset, max_steps=max_steps):
            max_step_failures += 1

    count = len(tasks)
    return {
        "dataset": exp.dataset,
        "graph": exp.graph,
        "memory": exp.memory,
        "tasks": count,
        "expected_tasks": exp.num_tasks,
        "em": sum(initial_em_values) / count if count else 0.0,
        "f1": sum(initial_f1_values) / count if count else 0.0,
        "retry_em": "",
        "retry_f1": "",
        "max_step_failures": max_step_failures,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "result_path": str(path.relative_to(ROOT)),
    }


def build_command(
    exp: Experiment,
    out_dir: Path,
    *,
    concurrency: int,
    max_steps: int,
    result_path: Path | None = None,
    task_indices: list[int] | None = None,
) -> list[str]:
    result_path = result_path or (out_dir / "result.jsonl")
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--dataset",
        exp.dataset,
        "--graph",
        exp.graph,
        "--model",
        "gpt-4o-mini",
        "--api-profile",
        "gpt-4o-mini",
        "--num-tasks",
        str(exp.num_tasks),
        "--concurrency",
        "1",
        "--max-steps",
        str(max_steps),
        "--result-path",
        str(result_path),
    ]
    if task_indices:
        cmd.append("--task-indices")
        cmd.extend(str(idx) for idx in task_indices)
    memory_dir = out_dir / "memory_store"
    cmd.extend(
        [
            "--mas-memory-backend",
            "c2mem",
            "--mas-memory-path",
            str(memory_dir),
            "--mas-memory-topk",
            "2",
        ]
    )
    return cmd


def _task_index(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("task_index"))
    except (TypeError, ValueError):
        return None


def _existing_task_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indices: set[int] = set()
    for row in _jsonl_rows(path):
        if row.get("type") != "task":
            continue
        idx = _task_index(row)
        if idx is not None:
            indices.add(idx)
    return indices


def _has_current_native_memory_result(path: Path) -> bool:
    if not path.exists():
        return False
    for row in _jsonl_rows(path):
        if row.get("type") != "config":
            continue
        args = row.get("args") or {}
        return args.get("mas_memory_implementation") == NATIVE_MAS_MEMORY_VERSION
    return False


def _native_store_matches_result(
    out_dir: Path,
    exp: Experiment,
    task_indices: set[int],
) -> bool:
    if not task_indices:
        return True
    if task_indices != set(range(len(task_indices))):
        return False
    state_path = out_dir / "memory_store" / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if state.get("backend") != EXTERNAL_BACKEND[exp.memory]:
        return False
    return int(state.get("record_count", -1) or 0) == len(task_indices)


def _merge_resume_result(result_path: Path, temp_result_path: Path, exp: Experiment) -> int:
    existing_rows = _jsonl_rows(result_path) if result_path.exists() else []
    temp_rows = _jsonl_rows(temp_result_path)

    config_rows = [row for row in existing_rows if row.get("type") == "config"]
    if not config_rows:
        config_rows = [row for row in temp_rows if row.get("type") == "config"][:1]

    passthrough_rows: list[dict[str, Any]] = []
    task_by_index: dict[int, dict[str, Any]] = {}
    for row in existing_rows + temp_rows:
        row_type = row.get("type")
        if row_type in {"config", "summary"}:
            continue
        if row_type != "task":
            passthrough_rows.append(row)
            continue
        idx = _task_index(row)
        if idx is None:
            passthrough_rows.append(row)
            continue
        task_by_index[idx] = row

    merged_rows = config_rows[:1] + passthrough_rows
    merged_rows.extend(task_by_index[idx] for idx in sorted(task_by_index))
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_rows),
        encoding="utf-8",
    )
    return len([idx for idx in task_by_index if 0 <= idx < exp.num_tasks])


def _remove_child_dir(parent: Path, child_name: str) -> None:
    child = (parent / child_name).resolve()
    parent_resolved = parent.resolve()
    if child == parent_resolved or parent_resolved not in child.parents:
        raise RuntimeError(f"refusing to remove path outside experiment dir: {child}")
    if child.exists():
        shutil.rmtree(child)


async def run_experiment(
    exp: Experiment,
    *,
    out_root: Path,
    semaphore: asyncio.Semaphore,
    concurrency: int,
    max_steps: int,
    rerun: bool,
) -> tuple[Experiment, int]:
    try:
        out_dir = out_root / exp.combo_id
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / "result.jsonl"
        uses_native_memory = exp.memory in EXTERNAL_BACKEND
        if (
            uses_native_memory
            and result_path.exists()
            and not rerun
            and not _has_current_native_memory_result(result_path)
        ):
            rerun = True
            append_runner_event(
                out_root,
                {"event": "invalidate_legacy_memory_result", "combo": exp.combo_id},
            )
        existing_indices: set[int] = set()
        missing_indices: list[int] = []
        command_result_path = result_path
        if result_path.exists() and not rerun:
            existing_indices = _existing_task_indices(result_path)
            if uses_native_memory and not _native_store_matches_result(
                out_dir,
                exp,
                existing_indices,
            ):
                rerun = True
                append_runner_event(
                    out_root,
                    {
                        "event": "invalidate_nonsequential_memory_resume",
                        "combo": exp.combo_id,
                        "existing_tasks": sorted(existing_indices),
                    },
                )
        if result_path.exists() and not rerun:
            missing_indices = sorted(set(range(exp.num_tasks)) - existing_indices)
            if not missing_indices:
                append_runner_event(out_root, {"event": "skip_complete", "combo": exp.combo_id, "tasks": len(existing_indices)})
                return exp, 0
            if existing_indices:
                command_result_path = out_dir / "resume_missing.jsonl"
                if command_result_path.exists():
                    command_result_path.unlink()
                append_runner_event(
                    out_root,
                    {
                        "event": "resume_partial",
                        "combo": exp.combo_id,
                        "existing_tasks": len(existing_indices),
                        "missing_tasks": missing_indices,
                    },
                )
        if rerun:
            _remove_child_dir(out_dir, "memory_store")

        cmd = build_command(
            exp,
            out_dir,
            concurrency=concurrency,
            max_steps=max_steps,
            result_path=command_result_path,
            task_indices=missing_indices,
        )
        command_file = "command.resume.json" if missing_indices else "command.json"
        (out_dir / command_file).write_text(json.dumps({"cmd": cmd}, ensure_ascii=False, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        async with semaphore:
            append_runner_event(out_root, {"event": "start", "combo": exp.combo_id, "cmd": cmd})
            stdout_name = "stdout.resume.log" if missing_indices else "stdout.log"
            stderr_name = "stderr.resume.log" if missing_indices else "stderr.log"
            with (out_dir / stdout_name).open("wb") as stdout, (out_dir / stderr_name).open("wb") as stderr:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(ROOT),
                    stdout=stdout,
                    stderr=stderr,
                    env=env,
                )
                append_runner_event(out_root, {"event": "process_started", "combo": exp.combo_id, "pid": process.pid})
                return_code = await process.wait()
            append_runner_event(out_root, {"event": "exit", "combo": exp.combo_id, "return_code": int(return_code)})
        if int(return_code) == 0 and missing_indices:
            merged_count = _merge_resume_result(result_path, command_result_path, exp)
            append_runner_event(out_root, {"event": "resume_merged", "combo": exp.combo_id, "tasks": merged_count})
        return exp, int(return_code)
    except BaseException as exc:
        append_runner_event(
            out_root,
            {
                "event": "runner_exception",
                "combo": exp.combo_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return exp, 999


def all_experiments(memories: list[str] | None = None) -> list[Experiment]:
    selected_memories = memories if memories is not None else MEMORIES
    return [
        Experiment(dataset=dataset, graph=graph, memory=memory, num_tasks=num_tasks)
        for dataset, num_tasks in DATASETS.items()
        for graph in GRAPHS
        for memory in selected_memories
    ]


def write_summary(
    out_root: Path,
    *,
    max_steps: int,
    memories: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in all_experiments(memories):
        result_path = out_root / exp.combo_id / "result.jsonl"
        if not result_path.exists():
            rows.append(
                {
                    "dataset": exp.dataset,
                    "graph": exp.graph,
                    "memory": exp.memory,
                    "tasks": 0,
                    "expected_tasks": exp.num_tasks,
                    "error": "missing result.jsonl",
                }
            )
            continue
        if exp.memory in EXTERNAL_BACKEND and not _has_current_native_memory_result(result_path):
            rows.append(
                {
                    "dataset": exp.dataset,
                    "graph": exp.graph,
                    "memory": exp.memory,
                    "tasks": 0,
                    "expected_tasks": exp.num_tasks,
                    "error": "legacy native-memory result; C2Mem rerun required",
                    "result_path": str(result_path.relative_to(ROOT)),
                }
            )
            continue
        row = summarize_result(result_path, exp, max_steps=max_steps)
        stderr_path = out_root / exp.combo_id / "stderr.log"
        if stderr_path.exists():
            row["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="ignore")[-1000:]
        rows.append(row)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_overall_tables(out_root, rows)
    return rows


def write_overall_tables(out_root: Path, rows: list[dict[str, Any]]) -> None:
    overall_rows: list[dict[str, Any]] = []
    for row in rows:
        tasks = int(row.get("tasks", 0) or 0)
        expected = int(row.get("expected_tasks", 0) or 0)
        status = "complete" if tasks == expected and expected > 0 else "partial" if tasks else "missing"

        def metric(name: str) -> str:
            value = row.get(name, "")
            return "" if value == "" or value is None else f"{float(value):.3f}"

        overall_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "mas": row.get("graph", ""),
                "memory": row.get("memory", ""),
                "status": status,
                "tasks": tasks,
                "expected_tasks": expected,
                "progress": f"{tasks}/{expected}",
                "em": metric("em"),
                "f1": metric("f1"),
                "retry_em": metric("retry_em"),
                "retry_f1": metric("retry_f1"),
                "max_step_failures": int(row.get("max_step_failures", 0) or 0),
                "prompt_tokens": int(row.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(row.get("completion_tokens", 0) or 0),
                "total_tokens": int(row.get("total_tokens", 0) or 0),
                "result_path": row.get("result_path", ""),
            }
        )

    json_path = out_root / "overall_summary_table.json"
    json_path.write_text(json.dumps(overall_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(overall_rows[0]) if overall_rows else []
    with (out_root / "overall_summary_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(overall_rows)

    total = len(overall_rows)
    complete = sum(row["status"] == "complete" for row in overall_rows)
    partial = sum(row["status"] == "partial" for row in overall_rows)
    lines = [
        "# Overall Result Summary - gpt-4o-mini",
        "",
        f"Progress: complete {complete}/{total}, partial {partial}/{total}, missing {total - complete - partial}/{total}",
        "",
        "| Dataset | MAS | Memory | Status | Progress | EM | F1 | Retry EM | Retry F1 | MaxStepFail | Tokens |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['dataset']} | {row['mas']} | {row['memory']} | {row['status']} | {row['progress']} | "
            f"{row['em']} | {row['f1']} | {row['retry_em']} | {row['retry_f1']} | "
            f"{row['max_step_failures']} | {row['total_tokens']} |"
        )
    (out_root / "overall_summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    leaderboard = ["# Complete Runs Leaderboard"]
    for dataset in DATASETS:
        leaderboard.extend(
            [
                "",
                f"## {dataset}",
                "",
                "| MAS | Memory | EM | F1 | Retry EM | Retry F1 | MaxStepFail | Tokens |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        dataset_rows = [
            row for row in overall_rows if row["dataset"] == dataset and row["status"] == "complete"
        ]
        dataset_rows.sort(
            key=lambda row: (-float(row["em"] or 0.0), -float(row["f1"] or 0.0), row["mas"], row["memory"])
        )
        for row in dataset_rows:
            leaderboard.append(
                f"| {row['mas']} | {row['memory']} | {row['em']} | {row['f1']} | "
                f"{row['retry_em']} | {row['retry_f1']} | {row['max_step_failures']} | {row['total_tokens']} |"
            )
    (out_root / "complete_leaderboard.md").write_text("\n".join(leaderboard) + "\n", encoding="utf-8")


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run C2Mem experiments across MAS graphs and datasets.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--experiment-parallelism", type=int, default=12)
    parser.add_argument("--task-concurrency", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--only-dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--only-graph", choices=GRAPHS, default=None)
    parser.add_argument("--only-memory", choices=MEMORY_CHOICES, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    out_root = args.out_dir.resolve()
    selected_memories = [args.only_memory] if args.only_memory else MEMORIES
    experiments = all_experiments(selected_memories)
    if args.only_dataset:
        experiments = [exp for exp in experiments if exp.dataset == args.only_dataset]
    if args.only_graph:
        experiments = [exp for exp in experiments if exp.graph == args.only_graph]
    async def run_batch(batch: list[Experiment]) -> None:
        semaphore = asyncio.Semaphore(max(1, int(args.experiment_parallelism)))
        tasks = [
            run_experiment(
                exp,
                out_root=out_root,
                semaphore=semaphore,
                concurrency=max(1, int(args.task_concurrency)),
                max_steps=max(1, int(args.max_steps)),
                rerun=bool(args.rerun),
            )
            for exp in batch
        ]
        for future in asyncio.as_completed(tasks):
            exp, code = await future
            print(f"{exp.combo_id}: exit={code}", flush=True)

    if not args.summary_only:
        await run_batch(experiments)

    rows = write_summary(
        out_root,
        max_steps=max(1, int(args.max_steps)),
        memories=selected_memories,
    )
    print(f"summary written: {out_root / 'summary.csv'} ({len(rows)} rows)")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
