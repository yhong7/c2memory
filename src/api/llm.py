from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Protocol

from loguru import logger

from src.configs import load_config

if TYPE_CHECKING:
    from .api_manager import APIRequestManager


CONFIG = load_config()
LLM_CONFIG = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMCallable(Protocol):
    async def __call__(
        self,
        messages: list[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[list[str]] = None,
        num_comps: int = NUM_COMPS,
    ) -> str:
        ...


class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def __call__(
        self,
        messages: list[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[list[str]] = None,
        num_comps: int = NUM_COMPS,
    ) -> str:
        ...


class GPTChat(LLM):
    def __init__(self, model_name: str, api_manager: "APIRequestManager"):
        super().__init__(model_name)
        self.api_manager = api_manager
        logger.debug(
            "GPTChat init: request_model={}, default_model={}, RPM={}, RPS={:.2f}",
            self.model_name,
            self.api_manager.model_name,
            self.api_manager.rpm,
            self.api_manager.rps,
        )

    async def __call__(
        self,
        messages: list[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[list[str]] = None,
        num_comps: int = NUM_COMPS,
    ) -> str:
        payload = [{"role": msg.role, "content": msg.content} for msg in messages]
        return await self.api_manager.chat_completion(
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
            model=self.model_name,
        )


