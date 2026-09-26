import asyncio
import sys

from src.main import main as cli_main


# Edit these values directly, then run:
#   conda run -n c2mem python run.py
CONFIG = {
    "dataset": "fever",
    "graph": "dylan",
    "model": None,
    "task": None,
    "subagents": 4,
    "dylan_max_layers": 5,
    "dylan_height": 3,
    "rounds": 1,
    "max_steps": 10,
    "num_tasks": 1,
    "mas_memory_backend": "c2mem",
    "mas_memory_path": ".db/c2mem/fever-dylan",
    "mas_memory_topk": 2,
    "mode": "train",
    "concurrency": 1,
}


def build_argv() -> list[str]:
    argv = ["run.py"]
    for key, value in CONFIG.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        argv.extend([flag, str(value)])
    return argv


if __name__ == "__main__":
    sys.argv = build_argv()
    asyncio.run(cli_main())
