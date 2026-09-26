from dataclasses import dataclass
import asyncio
import os
from typing import Any, ClassVar, Optional

from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, OpenAI

from .observability import observe
from src.configs import load_config


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first(*values: Any) -> Any:
    for value in values:
        cleaned = _clean(value)
        if cleaned is not None:
            return cleaned
    return None


def _env_value(name: Any) -> str | None:
    key = _clean(name)
    if not key:
        return None
    return _clean(os.getenv(key))


@dataclass
class APIRequestManager:
    global_token_usage_tree: ClassVar[dict[str, Any]] = {
        "name": "global",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "children": {},
    }

    api_profile_id: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    rpm: int | None = None
    rps: Optional[float] = None
    reasoning_effort: str | None = None
    thinking_enabled: bool | None = None
    embedding_api_base: str | None = None
    embedding_api_key: str | None = None
    embedding_model_name: str | None = None
    embedding_max_retries: int | None = None
    embedding_retry_base_delay: float | None = None
    chat_max_retries: int | None = None
    chat_retry_base_delay: float | None = None

    def __post_init__(self):
        config = load_config()
        api_config = config.get("api_config", {})
        llm_config = config.get("llm_config", {})
        profiles = config.get("api_profiles", {}) or {}
        requested_model = _clean(self.model_name)
        explicit_profile_id = _first(self.api_profile_id)
        if explicit_profile_id:
            profile_id = explicit_profile_id
        elif requested_model in profiles:
            # CLI/config callers often use a profile id as the model selector
            # (for example: --model gpt-4o-mini). Treat that as a provider
            # profile, not as a model sent to the default provider.
            profile_id = requested_model
            requested_model = None
        else:
            profile_id = _first(os.getenv("OPENAI_API_PROFILE_ID"), os.getenv("OPENAI_API_PROFILE"))
        if profile_id and profile_id not in profiles:
            raise ValueError(f"Unknown API profile id: {profile_id}")
        profile = dict(profiles.get(profile_id, {}) or {}) if profile_id else {}
        profile_key = _first(
            _env_value(profile.get("openai_api_key_env")),
            profile.get("openai_api_key"),
        )
        default_key = _first(
            _env_value(api_config.get("openai_api_key_env")),
            os.getenv("OPENAI_API_KEY"),
            api_config.get("openai_api_key"),
        )
        embedding_key = _first(
            _env_value(api_config.get("embedding_api_key_env")),
            os.getenv("EMBEDDING_API_KEY"),
            api_config.get("embedding_api_key"),
        )

        self.api_profile_id = profile_id
        if profile:
            self.api_base = _first(
                self.api_base,
                _env_value(profile.get("openai_api_base_env")),
                os.getenv("OPENAI_API_BASE"),
                profile.get("openai_api_base"),
                api_config.get("openai_api_base"),
            )
            self.api_key = _first(self.api_key, profile_key, default_key)
            self.model_name = _first(
                requested_model,
                profile.get("model_name"),
                api_config.get("model_name"),
                "gpt-3.5-turbo",
            )
        else:
            self.api_base = _first(self.api_base, os.getenv("OPENAI_API_BASE"), api_config.get("openai_api_base"))
            self.api_key = _first(self.api_key, default_key)
            self.model_name = _first(
                self.model_name,
                os.getenv("OPENAI_MODEL_NAME"),
                api_config.get("model_name"),
                "gpt-3.5-turbo",
            )
        if not self.api_base:
            raise RuntimeError("OpenAI-compatible chat API base URL is not configured")
        if not self.api_key:
            raise RuntimeError(
                f"OpenAI-compatible chat API key is not configured for profile {self.api_profile_id or '(default)'}"
            )
        self.rpm = int(self.rpm or api_config.get("rpm", 1000))
        self.rps = self.rps if self.rps is not None else api_config.get("rps") or self.rpm / 60.0
        self.reasoning_effort = self.reasoning_effort or llm_config.get("reasoning_effort", "low")
        self.thinking_enabled = bool(
            self.thinking_enabled if self.thinking_enabled is not None else llm_config.get("thinking_enabled", False)
        )
        self.embedding_api_base = _first(
            self.embedding_api_base,
            os.getenv("EMBEDDING_API_BASE"),
            api_config.get("embedding_api_base"),
            "",
        ) or ""
        self.embedding_api_key = _first(self.embedding_api_key, embedding_key, "") or ""
        self.embedding_model_name = _first(
            self.embedding_model_name,
            os.getenv("EMBEDDING_MODEL_NAME"),
            api_config.get("embedding_model_name"),
            "",
        ) or ""
        self.embedding_max_retries = int(self.embedding_max_retries or api_config.get("embedding_max_retries", 3))
        self.embedding_retry_base_delay = float(
            self.embedding_retry_base_delay or api_config.get("embedding_retry_base_delay", 1.0)
        )
        self.chat_max_retries = int(
            self.chat_max_retries or os.getenv("OPENAI_CHAT_MAX_RETRIES") or api_config.get("chat_max_retries", 3)
        )
        self.chat_retry_base_delay = float(
            self.chat_retry_base_delay
            or os.getenv("OPENAI_CHAT_RETRY_BASE_DELAY")
            or api_config.get("chat_retry_base_delay", 1.0)
        )
        self.async_client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        self.embedding_client: OpenAI | None = None
        self.embedding_async_client: AsyncOpenAI | None = None
        if self.embedding_api_base and self.embedding_api_key and self.embedding_model_name:
            self.embedding_async_client = AsyncOpenAI(
                base_url=self._openai_base_url(self.embedding_api_base),
                api_key=self.embedding_api_key,
            )
        self._shared_clients = {
            "async_client": self.async_client,
            "embedding_client": self.embedding_client,
            "embedding_async_client": self.embedding_async_client,
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._usage_node_path = [self.global_token_usage_tree]
        self._task_usage_state = {"total_tokens": 0}

    def resolved_chat_config(self) -> dict[str, str | None]:
        return {
            "api_profile_id": self.api_profile_id,
            "api_base": self.api_base,
            "model_name": self.model_name,
        }

    @property
    def current_task_token_usage(self) -> int:
        return int(getattr(self, "_task_usage_state", {}).get("total_tokens", 0))

    @staticmethod
    def _new_usage_node(name: str) -> dict[str, Any]:
        return {
            "name": name,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "children": {},
        }

    def _child_usage_node(self, name: str) -> dict[str, Any]:
        parent = self._usage_node_path[-1] if getattr(self, "_usage_node_path", None) else self.global_token_usage_tree
        children = parent.setdefault("children", {})
        if name not in children:
            children[name] = self._new_usage_node(name)
        return children[name]

    def _current_usage_node(self) -> dict[str, Any]:
        path = getattr(self, "_usage_node_path", None)
        if path:
            return path[-1]
        return self.global_token_usage_tree

    def raise_child(self, name: str | None = None, *, new_task_scope: bool = False) -> "APIRequestManager":
        """Create a lightweight child manager sharing clients/config but writing to a child usage node.

        ``new_task_scope`` is intended for task-level children. Nested children
        inside a task should keep the default so their token usage contributes
        to the same task-level counter.
        """
        label = str(name or f"child-{len(getattr(self, '_usage_node_path', []))}").strip() or "child"
        child = object.__new__(self.__class__)
        child.__dict__.update(self.__dict__)
        child.prompt_tokens = 0
        child.completion_tokens = 0
        child._usage_node_path = list(getattr(self, "_usage_node_path", [self.global_token_usage_tree])) + [
            self._child_usage_node(label)
        ]
        child._task_usage_state = {"total_tokens": 0} if new_task_scope else getattr(
            self,
            "_task_usage_state",
            {"total_tokens": 0},
        )
        return child

    def _record_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        prompt = int(prompt_tokens or 0)
        completion = int(completion_tokens or 0)
        total = prompt + completion
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        for node in getattr(self, "_usage_node_path", [self.global_token_usage_tree]):
            node["prompt_tokens"] = int(node.get("prompt_tokens", 0)) + prompt
            node["completion_tokens"] = int(node.get("completion_tokens", 0)) + completion
            node["total_tokens"] = int(node.get("total_tokens", 0)) + total
        state = getattr(self, "_task_usage_state", None)
        if state is not None:
            state["total_tokens"] = int(state.get("total_tokens", 0)) + total

    @staticmethod
    def _supports_reasoning_effort(model: str) -> bool:
        return bool(model) and model.startswith(("gpt-5", "o1", "o3", "o4"))

    @staticmethod
    def _supports_thinking(model: str) -> bool:
        """DeepSeek models support a top-level ``thinking`` param."""
        return bool(model) and model.startswith("deepseek")

    @staticmethod
    def _is_retryable_api_error(exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
        if isinstance(exc, APIStatusError):
            status_code = exc.status_code
            return status_code == 429 or 500 <= status_code < 600
        return False

    @staticmethod
    def _openai_base_url(base_url: str) -> str:
        base = str(base_url or "").strip().rstrip("/")
        if base.endswith("/embeddings"):
            return base.rsplit("/", 1)[0]
        return base

    async def embed_texts(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        clean = [str(text).strip() for text in texts if str(text).strip()]
        if not clean:
            return []
        if self.embedding_async_client is None:
            raise RuntimeError("embedding API is not configured; refusing to return empty embeddings")
        model_name = model or self.embedding_model_name
        last_exc: Exception | None = None
        for attempt in range(max(1, self.embedding_max_retries)):
            try:
                response = await self._shared_clients["embedding_async_client"].embeddings.create(
                    model=model_name,
                    input=clean,
                )
                data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
                return [list(item.embedding) for item in data]
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable_api_error(exc) or attempt >= self.embedding_max_retries - 1:
                    raise
                await asyncio.sleep(self.embedding_retry_base_delay * (2 ** attempt))
        if last_exc is not None:
            raise last_exc
        return []

    def embed_texts_sync(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        clean = [str(text).strip() for text in texts if str(text).strip()]
        if not clean:
            return []
        if not self.embedding_api_base or not self.embedding_api_key or not self.embedding_model_name:
            raise RuntimeError("embedding API is not configured; refusing to return empty embeddings")
        if self._shared_clients["embedding_client"] is None:
            self._shared_clients["embedding_client"] = OpenAI(
                base_url=self._openai_base_url(self.embedding_api_base),
                api_key=self.embedding_api_key,
            )
        self.embedding_client = self._shared_clients["embedding_client"]
        model_name = model or self.embedding_model_name
        last_exc: Exception | None = None
        for attempt in range(max(1, self.embedding_max_retries)):
            try:
                response = self._shared_clients["embedding_client"].embeddings.create(
                    model=model_name,
                    input=clean,
                )
                data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
                return [list(item.embedding) for item in data]
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable_api_error(exc) or attempt >= self.embedding_max_retries - 1:
                    raise
                import time

                time.sleep(self.embedding_retry_base_delay * (2 ** attempt))
        if last_exc is not None:
            raise last_exc
        return []

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> str:
        model_name = model or self.model_name
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort and self._supports_reasoning_effort(model_name):
            kwargs["reasoning_effort"] = self.reasoning_effort

        if self._supports_thinking(model_name):
            kwargs["extra_body"] = {"thinking": {"type": "enabled" if self.thinking_enabled else "disabled"}}

        with observe(
            "openai-chat-completion",
            as_type="generation",
            input={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            metadata={"provider": "openai-compatible"},
            model=model_name,
        ) as generation:
            last_exc: Exception | None = None
            for attempt in range(max(1, self.chat_max_retries)):
                try:
                    response = await self._shared_clients["async_client"].chat.completions.create(**kwargs)
                    break
                except Exception as exc:
                    last_exc = exc
                    if not self._is_retryable_api_error(exc) or attempt >= self.chat_max_retries - 1:
                        raise
                    await asyncio.sleep(self.chat_retry_base_delay * (2 ** attempt))
            else:
                assert last_exc is not None
                raise last_exc
            usage = response.usage
            if usage is not None:
                self._record_token_usage(usage.prompt_tokens, usage.completion_tokens)
            content = response.choices[0].message.content or ""
            if generation is not None:
                generation.update(
                    output={"content": content},
                    usage_details={
                        "input": usage.prompt_tokens if usage else 0,
                        "output": usage.completion_tokens if usage else 0,
                    },
                )
            return content

    def get_token_usage(self) -> dict[str, int]:
        node = self._current_usage_node()
        return {
            "prompt_tokens": int(node.get("prompt_tokens", 0)),
            "completion_tokens": int(node.get("completion_tokens", 0)),
        }

    def get_local_token_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
