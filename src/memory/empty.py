from .base import MemoryRetrieverBase, MemoryUpdaterBase


class EmptyMemoryRetriever(MemoryRetrieverBase):
    def retrieve(self, agent_name: str, task: str) -> list[str]:
        return []


class EmptyMemoryUpdater(MemoryUpdaterBase):
    def update(self, agent_name: str, task: str, output: str) -> None:
        return None
