from __future__ import annotations

import argparse
import bz2
import itertools
import json
import re
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from time import perf_counter
from typing import Iterator
from urllib.parse import unquote

from src.configs import load_config
from src.log import logger, setup
from src.memory import ChromaVectorDB, OpenAICompatibleEmbedding


_ANCHOR_RE = re.compile(r'<a href="([^"]+)">(.+?)</a>')
DEFAULT_ARCHIVE = (
    "data/wikipedia/"
    "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
)
DEFAULT_COLLECTION = "wiki_abstracts"
DEFAULT_CHECKPOINT = "temp/wiki_abstracts_ingest_ckpt.json"
DEFAULT_LOG = "temp/wiki_abstracts_ingest.log"
DEFAULT_EXPECTED_TOTAL = 4_455_000
DEFAULT_BATCH_SIZE = 5000


@dataclass
class HotpotAbstractRecord:
    id: str
    url: str
    title: str
    text: list[str]
    charoffset: list
    text_with_links: list[str]
    charoffset_with_links: list


def _normalize_title(title: str) -> str:
    return " ".join(title.strip().lower().split())


def _extract_links(text_with_links: list[str]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for passage in text_with_links:
        for href, anchor in _ANCHOR_RE.findall(passage):
            target_title = unquote(unescape(href)).replace("_", " ")
            links.append(
                {
                    "anchor": unescape(anchor),
                    "href": href,
                    "target_title": target_title,
                    "target_title_norm": _normalize_title(target_title),
                }
            )
    return links


def build_chunk(record: HotpotAbstractRecord) -> str:
    return f"{record.title}: {' '.join(record.text)}"


def build_chroma_record(record: HotpotAbstractRecord) -> dict:
    chunk = build_chunk(record)
    return {
        "id": record.id,
        "url": record.url,
        "title": record.title,
        "text": json.dumps(record.text, ensure_ascii=False),
        "charoffset": json.dumps(record.charoffset, ensure_ascii=False),
        "text_with_links": json.dumps(record.text_with_links, ensure_ascii=False),
        "charoffset_with_links": json.dumps(record.charoffset_with_links, ensure_ascii=False),
        "title_norm": _normalize_title(record.title),
        "links": json.dumps(_extract_links(record.text_with_links), ensure_ascii=False),
        "chunk": chunk,
    }


def iter_abstract_records(
    archive_path: str | Path,
    *,
    start_member_index: int = 0,
    start_line_index: int = 0,
) -> Iterator[tuple[HotpotAbstractRecord, int, str, int]]:
    path = Path(archive_path)
    with tarfile.open(path, "r:bz2") as archive:
        member_index = -1
        for member in archive:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            member_index += 1
            if member_index < start_member_index:
                continue
            nested = archive.extractfile(member)
            if nested is None:
                continue
            with bz2.BZ2File(nested, "rb") as decompressed:
                for line_index, line in enumerate(decompressed):
                    if member_index == start_member_index and line_index < start_line_index:
                        continue
                    if not line:
                        continue
                    payload = json.loads(line)
                    yield (
                        HotpotAbstractRecord(
                            id=payload["id"],
                            url=payload["url"],
                            title=payload["title"],
                            text=payload["text"],
                            charoffset=payload["charoffset"],
                            text_with_links=payload["text_with_links"],
                            charoffset_with_links=payload["charoffset_with_links"],
                        ),
                        member_index,
                        member.name,
                        line_index,
                    )


def ensure_collection(db: ChromaVectorDB, collection: str, dim: int) -> None:
    client = db._client
    assert client is not None
    existing = {c.name for c in client.list_collections()}
    if collection in existing:
        return
    db.create_collection(collection, dim=dim)


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(log_path: str | Path, message: str) -> None:
    target = _ensure_parent(log_path)
    line = f"[{_utc_now()}] {message}"
    logger.info(line)
    with open(target, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def _load_checkpoint(path: str | Path) -> dict | None:
    target = Path(path)
    if not target.exists():
        return None
    with open(target, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_checkpoint(path: str | Path, payload: dict) -> None:
    target = _ensure_parent(path)
    with open(target, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _format_rate(count: int, seconds: float) -> str:
    if seconds <= 0:
        return "inf"
    return f"{count / seconds:.2f}"


def _format_eta(remaining: int, rate: float) -> str:
    if remaining <= 0:
        return "00:00:00"
    if rate <= 0:
        return "unknown"
    total_seconds = int(remaining / rate)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _make_checkpoint(
    *,
    archive_path: str | Path,
    collection: str,
    processed_count: int,
    member_index: int,
    member_name: str,
    line_index: int,
    last_id: str,
    batch_size: int,
    expected_total: int | None,
) -> dict:
    return {
        "updated_at_utc": _utc_now(),
        "archive_path": str(archive_path),
        "collection": collection,
        "processed_count": processed_count,
        "member_index": member_index,
        "member_name": member_name,
        "line_index": line_index,
        "last_id": last_id,
        "batch_size": batch_size,
        "expected_total": expected_total,
    }


def ingest_records(
    archive_path: str | Path,
    *,
    collection: str = DEFAULT_COLLECTION,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    log_path: str = DEFAULT_LOG,
    expected_total: int | None = DEFAULT_EXPECTED_TOTAL,
) -> dict:
    config = load_config()
    api_cfg = config["api_config"]
    embedding = OpenAICompatibleEmbedding(
        api_key=api_cfg["embedding_api_key"],
        model=api_cfg["embedding_model_name"],
        base_url=api_cfg["embedding_api_base"],
        batch_size=32,
        max_workers=4,
    )
    db = ChromaVectorDB(persist_dir=".db/chroma", embedding=embedding)
    ensure_collection(db, collection, embedding.dim)

    client = db._client
    assert client is not None
    existing_count = client.get_collection(collection).count()
    checkpoint = _load_checkpoint(checkpoint_path)

    skip_records = 0
    start_member_index = 0
    start_line_index = 0
    processed_count = 0

    if checkpoint is not None:
        processed_count = int(checkpoint["processed_count"])
        start_member_index = int(checkpoint["member_index"])
        start_line_index = int(checkpoint["line_index"]) + 1
        if existing_count < processed_count:
            raise RuntimeError(
                f"Collection count {existing_count} is smaller than checkpoint processed_count {processed_count}."
            )
        _log(
            log_path,
            f"resume from checkpoint: processed={processed_count}, member={checkpoint['member_name']}, line={checkpoint['line_index']}, existing_count={existing_count}",
        )
    elif existing_count > 0:
        skip_records = existing_count
        processed_count = existing_count
        _log(
            log_path,
            f"no checkpoint found; resume from existing collection count={existing_count} to avoid duplicate test data",
        )
    else:
        _log(log_path, "start fresh ingest from record 0")

    max_new_records = None if limit is None or limit <= 0 else limit
    initial_processed_count = processed_count
    _log(
        log_path,
        f"config: collection={collection}, archive={archive_path}, batch_size={batch_size}, limit={max_new_records or 'FULL'}, expected_total={expected_total or 'unknown'}",
    )

    started_at = perf_counter()
    batch_started_at = perf_counter()
    total_new_records = 0
    batch_records: list[dict] = []
    last_position: tuple[int, str, int, str] | None = None

    iterator = iter_abstract_records(
        archive_path,
        start_member_index=start_member_index,
        start_line_index=start_line_index,
    )
    if skip_records > 0:
        iterator = itertools.islice(iterator, skip_records, None)

    def flush_batch() -> None:
        nonlocal batch_records
        nonlocal total_new_records
        nonlocal processed_count
        nonlocal batch_started_at
        if not batch_records or last_position is None:
            return

        write_started_at = perf_counter()
        db.upsert(collection, batch_records, embedding_field="chunk")
        write_seconds = perf_counter() - write_started_at

        batch_len = len(batch_records)
        total_new_records += batch_len
        processed_count += batch_len

        member_index, member_name, line_index, last_id = last_position
        checkpoint_payload = _make_checkpoint(
            archive_path=archive_path,
            collection=collection,
            processed_count=processed_count,
            member_index=member_index,
            member_name=member_name,
            line_index=line_index,
            last_id=last_id,
            batch_size=batch_size,
            expected_total=expected_total,
        )
        _save_checkpoint(checkpoint_path, checkpoint_payload)

        batch_seconds = perf_counter() - batch_started_at
        total_seconds = perf_counter() - started_at
        remaining = max((expected_total or 0) - processed_count, 0)
        current_run_processed = processed_count - initial_processed_count
        overall_rate = current_run_processed / total_seconds if total_seconds > 0 else 0.0
        _log(
            log_path,
            "batch committed: "
            f"size={batch_len}, processed_total={processed_count}, last_id={last_id}, "
            f"batch_s={batch_seconds:.2f}, write_s={write_seconds:.2f}, "
            f"batch_rate={_format_rate(batch_len, batch_seconds)} docs/s, "
            f"overall_rate={overall_rate:.2f} docs/s, eta={_format_eta(remaining, overall_rate)}",
        )
        batch_records = []
        batch_started_at = perf_counter()

    for record, member_index, member_name, line_index in iterator:
        batch_records.append(build_chroma_record(record))
        last_position = (member_index, member_name, line_index, record.id)

        if len(batch_records) >= batch_size:
            flush_batch()

        if max_new_records is not None and total_new_records + len(batch_records) >= max_new_records:
            break

    if max_new_records is not None and batch_records:
        remaining_limit = max_new_records - total_new_records
        if remaining_limit < len(batch_records):
            batch_records = batch_records[:remaining_limit]

    flush_batch()

    total_seconds = perf_counter() - started_at
    current_run_processed = processed_count - initial_processed_count
    _log(
        log_path,
        f"ingest finished: new_records={total_new_records}, processed_total={processed_count}, total_s={total_seconds:.2f}, overall_rate={_format_rate(current_run_processed, total_seconds)} docs/s",
    )
    return {
        "new_records": total_new_records,
        "processed_total": processed_count,
        "total_seconds": round(total_seconds, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest HotpotQA wiki abstracts into ChromaDB.")
    parser.add_argument("--archive", type=str, default=DEFAULT_ARCHIVE)
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION)
    parser.add_argument("--limit", type=int, default=0, help="0 means full ingest")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--log-path", type=str, default=DEFAULT_LOG)
    parser.add_argument("--expected-total", type=int, default=DEFAULT_EXPECTED_TOTAL)
    args = parser.parse_args()
    setup("hotpotqa_abstracts_ingest")

    summary = ingest_records(
        args.archive,
        collection=args.collection,
        limit=args.limit,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        log_path=args.log_path,
        expected_total=args.expected_total,
    )
    logger.info("Ingest summary:\n{}", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
