# Docker usage

This repository is primarily a batch CLI application. The container runs
`python -m src.main` by default and writes state to mounted host directories.

## Build

```bash
docker compose build c2mem
```

## Configure

```bash
cp .env.example .env
```

Fill both API base URL/key pairs and set `WIKI_SERVICE_URL` to your own
deployment. The `.env` file is mounted
read-only into the container and is excluded from the image build context.
Secrets are intentionally not passed through `env_file`, because
`docker compose config` expands `env_file` values into its output.

External services must be reachable from inside the container. Their addresses
are deployment-specific and are not supplied by this repository.

## Run one default HotpotQA C2Mem task

```bash
docker compose run --rm c2mem
```

## Override runtime parameters

```bash
DATASET=hotpotqa NUM_TASKS=5 MAX_STEPS=8 CONCURRENCY=1 \
MAS_MEMORY_BACKEND=c2mem MAS_MEMORY_PATH=.db/c2mem/hotpotqa-autogen GRAPH=autogen \
docker compose run --rm c2mem
```

C2Mem requires task-level concurrency `1` so every task observes all posterior
updates from earlier tasks in a deterministic order.

## Interactive shell

```bash
docker compose --profile dev run --rm shell
```

Mounted directories:

- `.db` -> `/app/.db`
- `data` -> `/app/data`
- `logs` -> `/app/logs`
- `result` -> `/app/result`
- `temp` -> `/app/temp`
