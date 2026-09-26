from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ServiceSettings:
    host: str = field(default_factory=lambda: os.getenv("WIKI_SERVICE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("WIKI_SERVICE_PORT", "8000")))
    chroma_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("WIKI_SERVICE_CHROMA_DIR", "data/chroma")
        ).resolve()
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("WIKI_SERVICE_COLLECTION", "wiki_abstracts")
    )
    default_top_k: int = field(
        default_factory=lambda: int(os.getenv("WIKI_SERVICE_DEFAULT_TOP_K", "10"))
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("WIKI_SERVICE_TIMEOUT_SECONDS", "60"))
    )
    warmup_query: str = field(
        default_factory=lambda: os.getenv("WIKI_SERVICE_WARMUP_QUERY", "Albert Einstein")
    )

    embed_mode: str = field(
        default_factory=lambda: os.getenv("WIKI_SERVICE_EMBED_MODE", "api").strip().lower()
    )
    embedding_api_base: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_BASE", "").strip()
    )
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "").strip()
    )
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "").strip()
    )
    local_embedding_model: str = field(
        default_factory=lambda: os.getenv("WIKI_SERVICE_LOCAL_EMBED_MODEL", "").strip()
    )

    def validate(self) -> None:
        if self.default_top_k < 1:
            raise ValueError("WIKI_SERVICE_DEFAULT_TOP_K must be at least 1")
        if self.embed_mode == "api":
            missing = [
                name
                for name, value in (
                    ("EMBEDDING_API_BASE", self.embedding_api_base),
                    ("EMBEDDING_API_KEY", self.embedding_api_key),
                    ("EMBEDDING_MODEL_NAME", self.embedding_model_name),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "API embedding mode requires: " + ", ".join(missing)
                )
        elif self.embed_mode == "local":
            if not self.local_embedding_model:
                raise ValueError(
                    "WIKI_SERVICE_LOCAL_EMBED_MODEL is required for local embedding mode"
                )
            # A local path is required so SentenceTransformer cannot interpret
            # the value as a remote model-repository identifier.
            if not Path(self.local_embedding_model).expanduser().is_dir():
                raise ValueError(
                    "WIKI_SERVICE_LOCAL_EMBED_MODEL must point to an existing local directory"
                )
        else:
            raise ValueError("WIKI_SERVICE_EMBED_MODE must be 'api' or 'local'")


def load_settings(*, validate: bool = True) -> ServiceSettings:
    settings = ServiceSettings()
    if validate:
        settings.validate()
    return settings
