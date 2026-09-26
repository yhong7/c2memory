from dataclasses import dataclass

from src.api.llm import GPTChat, Message


@dataclass
class BaseAgent:
    name: str
    system_prompt: str
    llm: GPTChat

    async def respond(self, user_prompt: str) -> str:
        messages = [
            Message("system", self.system_prompt),
            Message("user", user_prompt),
        ]
        return (await self.llm(messages)).strip()
