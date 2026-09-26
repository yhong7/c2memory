from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


# This reader is local-only by design so that dataset acquisition and index
# construction remain explicit, reproducible steps.
def iter_embedding_records(dataset_dir: Path) -> Iterable[dict]:
    parquet_files = sorted(dataset_dir.rglob("*.parquet"))
    jsonl_files = sorted(dataset_dir.rglob("*.jsonl"))
    if parquet_files:
        from pyarrow import parquet as pq

        for file in parquet_files:
            yield from pq.read_table(file).to_pylist()
        return
    if jsonl_files:
        for file in jsonl_files:
            with file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        return
    raise FileNotFoundError(f"No parquet or JSONL files found under {dataset_dir}")


def _embedding(row: dict) -> list[float]:
    value = row.get("embedding")
    if value is None:
        raise ValueError("Each row must contain an embedding field")
    if isinstance(value, str):
        value = json.loads(value)
    return [float(item) for item in value]


def _metadata(row: dict) -> dict:
    metadata = {}
    for key, value in row.items():
        if key in {"embedding", "chunk", "document"} or value is None:
            continue
        metadata[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma from embedded records")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--collection", default="wiki_abstracts")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    client = chromadb.PersistentClient(
        path=str(Path(args.target_dir).resolve()),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass
    collection = client.create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    batch: list[dict] = []
    for index, row in enumerate(iter_embedding_records(Path(args.snapshot_dir).resolve())):
        document = row.get("chunk") or row.get("document")
        if document is None:
            raise ValueError("Each row must contain a chunk or document field")
        batch.append(
            {
                "id": str(row.get("id", index)),
                "document": str(document),
                "metadata": _metadata(row),
                "embedding": _embedding(row),
            }
        )
        if len(batch) >= args.batch_size:
            _upsert(collection, batch)
            batch = []
        if args.limit > 0 and index + 1 >= args.limit:
            break
    if batch:
        _upsert(collection, batch)


def _upsert(collection, records: list[dict]) -> None:
    collection.upsert(
        ids=[record["id"] for record in records],
        documents=[record["document"] for record in records],
        metadatas=[record["metadata"] for record in records],
        embeddings=[record["embedding"] for record in records],
    )


if __name__ == "__main__":
    main()
