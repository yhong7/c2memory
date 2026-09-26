from typing import Protocol


class MemoryProtocol(Protocol):
    def retrieve(self, agent_name: str, task: str) -> list[str]:
        ...

    def update(self, agent_name: str, task: str, output: str) -> None:
        ...


class AgentProtocol(Protocol):
    name: str

    async def respond(self, user_prompt: str) -> str:
        ...
