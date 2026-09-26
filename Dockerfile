# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# build-essential is needed by a few scientific Python wheels when a prebuilt
# wheel is unavailable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY src ./src
COPY run.py README.md ./
COPY docker ./docker

RUN mkdir -p /app/.db /app/data /app/logs /app/result /app/temp \
    && chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["/bin/sh", "/app/docker/entrypoint.sh"]
CMD ["python", "-m", "src.main", "--dataset", "hotpotqa", "--graph", "dylan", "--mas-memory-backend", "c2mem", "--mas-memory-path", ".db/c2mem/hotpotqa-dylan", "--num-tasks", "1", "--max-steps", "8", "--concurrency", "1"]
