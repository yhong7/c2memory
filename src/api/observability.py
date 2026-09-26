import os
from contextlib import contextmanager
from typing import Any

from loguru import logger

_LANGFUSE_INITIALIZED = False
_LANGFUSE_ENABLED = False


def _has_langfuse_config() -> bool:
    return all(
        os.getenv(key)
        for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    )


def init_langfuse() -> bool:
    global _LANGFUSE_INITIALIZED, _LANGFUSE_ENABLED

    if _LANGFUSE_INITIALIZED:
        return _LANGFUSE_ENABLED

    _LANGFUSE_INITIALIZED = True
    if not _has_langfuse_config():
        return False

    try:
        from langfuse import Langfuse

        Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_BASE_URL"),
        )
        _LANGFUSE_ENABLED = True
    except Exception as exc:
        logger.warning("Failed to initialize Langfuse: {}", exc)
        _LANGFUSE_ENABLED = False

    return _LANGFUSE_ENABLED


@contextmanager
def observe(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    output: Any = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
    trace_context: dict[str, str] | None = None,
):
    if not init_langfuse():
        yield None
        return

    from langfuse import get_client

    langfuse = get_client()
    kwargs: dict[str, Any] = {
        "as_type": as_type,
        "name": name,
        "input": input,
        "output": output,
        "metadata": metadata,
        "model": model,
    }

    if trace_context is not None:
        # Explicit parent for threads – also set as current so nested
        # observations (e.g. chat-completion generations) nest correctly.
        with langfuse.start_as_current_observation(**kwargs, trace_context=trace_context) as observation:
            yield observation
    else:
        with langfuse.start_as_current_observation(**kwargs) as observation:
            yield observation


@contextmanager
def propagate_trace_attributes(*, trace_name: str | None = None):
    if not init_langfuse() or not trace_name:
        yield
        return

    from langfuse import propagate_attributes

    with propagate_attributes(trace_name=trace_name):
        yield


def flush_langfuse() -> None:
    if not init_langfuse():
        return

    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        logger.warning("Failed to flush Langfuse events: {}", exc)
