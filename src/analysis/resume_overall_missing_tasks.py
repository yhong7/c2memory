from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from src.analysis.run_overall_4omini_experiments import (
    DATASETS,
    DEFAULT_OUT_DIR,
    GRAPHS,
    MEMORIES,
    Experiment,
    build_command,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _task_index(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("task_index"))
    except (TypeError, ValueError):
        return None


def existing_task_indices(path: Path) -> set[int]:
    return {
        idx
        for row in _read_jsonl(path)
        if row.get("type") == "task"
        for idx in [_task_index(row)]
        if idx is not None
    }


def all_experiments() -> list[Experiment]:
    return [
        Experiment(dataset=dataset, graph=graph, memory=memory, num_tasks=num_tasks)
        for dataset, num_tasks in DATASETS.items()
        for graph in GRAPHS
        for memory in MEMORIES
    ]


def selected_experiments(args: argparse.Namespace) -> list[Experiment]:
    exps = all_experiments()
    if args.only_dataset:
        exps = [exp for exp in exps if exp.dataset == args.only_dataset]
    if args.only_graph:
        exps = [exp for exp in exps if exp.graph == args.only_graph]
    if args.only_memory:
        exps = [exp for exp in exps if exp.memory == args.only_memory]
    return exps


def missing_indices(out_root: Path, exp: Experiment) -> list[int]:
    result_path = out_root / exp.combo_id / "result.jsonl"
    have = existing_task_indices(result_path)
    return [idx for idx in range(exp.num_tasks) if idx not in have]


def merge_resume_file(out_root: Path, exp: Experiment, resume_path: Path) -> int:
    if not resume_path.exists():
        return 0
    result_path = out_root / exp.combo_id / "result.jsonl"
    original_rows = _read_jsonl(result_path)
    resume_rows = _read_jsonl(resume_path)
    config_rows = [row for row in original_rows if row.get("type") == "config"]
    if not config_rows:
        config_rows = [row for row in resume_rows if row.get("type") == "config"][:1]
    task_by_index: dict[int, dict[str, Any]] = {}
    for row in original_rows:
        if row.get("type") == "task":
            idx = _task_index(row)
            if idx is not None:
                task_by_index[idx] = row
    merged_count = 0
    for row in resume_rows:
        if row.get("type") != "task":
            continue
        idx = _task_index(row)
        if idx is None:
            continue
        task_by_index[idx] = row
        merged_count += 1
    merged_tasks = [task_by_index[idx] for idx in sorted(task_by_index)]
    merged_rows = list(config_rows[:1]) + merged_tasks
    merged_rows.append(
        {
            "type": "summary",
            "merged": True,
            "tasks": len(merged_tasks),
            "expected_tasks": exp.num_tasks,
            "source_resume_file": str(resume_path),
        }
    )
    if result_path.exists():
        backup = result_path.with_suffix(".jsonl.bak")
        if not backup.exists():
            shutil.copy2(result_path, backup)
    _write_jsonl(result_path, merged_rows)
    return merged_count


async def run_command(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("OPENAI_CHAT_MAX_RETRIES", "8")
    env.setdefault("OPENAI_CHAT_RETRY_BASE_DELAY", "3")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        return int(await proc.wait())


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Resume missing task indices for overall experiments.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--task-concurrency", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--only-dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--only-graph", choices=GRAPHS, default=None)
    parser.add_argument("--only-memory", choices=MEMORIES, default=None)
    parser.add_argument("--max-combos", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    out_root = args.out_dir.resolve()
    exps = selected_experiments(args)
    runnable: list[tuple[Experiment, list[int]]] = []
    for exp in exps:
        missing = missing_indices(out_root, exp)
        if missing:
            runnable.append((exp, missing))
    if args.max_combos:
        runnable = runnable[: max(0, int(args.max_combos))]

    if args.merge_only:
        total = 0
        for exp, _ in runnable:
            total += merge_resume_file(out_root, exp, out_root / exp.combo_id / "resume_missing.jsonl")
        print(f"merged_task_rows={total}")
        return

    semaphore = asyncio.Semaphore(max(1, int(args.parallelism)))

    async def run_one(exp: Experiment, indices: list[int]) -> tuple[str, int]:
        out_dir = out_root / exp.combo_id
        resume_path = out_dir / "resume_missing.jsonl"
        cmd = build_command(
            exp,
            out_dir,
            concurrency=max(1, int(args.task_concurrency)),
            max_steps=max(1, int(args.max_steps)),
        )
        result_flag = cmd.index("--result-path")
        cmd[result_flag + 1] = str(resume_path)
        num_flag = cmd.index("--num-tasks")
        cmd[num_flag + 1] = str(exp.num_tasks)
        if "--task-indices" in cmd:
            raise RuntimeError("unexpected existing --task-indices in command")
        cmd.extend(["--task-indices", *[str(idx) for idx in indices]])
        (out_dir / "resume_command.json").write_text(
            json.dumps({"cmd": cmd, "missing_indices": indices}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        async with semaphore:
            code = await run_command(
                cmd,
                ROOT,
                out_dir / "resume_stdout.log",
                out_dir / "resume_stderr.log",
            )
            if code == 0:
                merge_resume_file(out_root, exp, resume_path)
            return exp.combo_id, code

    tasks = [run_one(exp, indices) for exp, indices in runnable]
    for future in asyncio.as_completed(tasks):
        combo, code = await future
        print(f"{combo}: exit={code}", flush=True)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
