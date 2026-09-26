from abc import ABC, abstractmethod
class MemoryStageBase(ABC):
    pass


class MemoryRetrieverBase(MemoryStageBase):
    @abstractmethod
    def retrieve(self, agent_name: str, task: str) -> list[str]:
        ...


class MemoryUpdaterBase(MemoryStageBase):
    @abstractmethod
    def update(self, agent_name: str, task: str, output: str) -> None:
        ...
