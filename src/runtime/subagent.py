from dataclasses import dataclass

from src.agents import SubAgentCoreAgent
from src.memory import MemoryRetrieverBase, MemoryUpdaterBase
from src.runtime.agent_node import AgentNodeBase
from src.runtime.prompts import build_subagent_core_prompt


@dataclass
class SubAgentRuntime(AgentNodeBase):
    core_agent: SubAgentCoreAgent
    retriever: MemoryRetrieverBase
    updater: MemoryUpdaterBase

    async def run(
        self,
        task: str,
        supervisor_brief: str,
        *,
        external_memory_context: list[str] | None = None,
    ) -> tuple[list[str], str]:
        return await self.run_node(
            task=task,
            prompt_builder=lambda memory_context: self._build_core_prompt(
                task,
                supervisor_brief,
                [*(external_memory_context or []), *memory_context],
            ),
            root_span_name="subagent-run",
            retrieve_span_name="subagent-memory-retrieve",
            execute_span_name="subagent-core-execution",
            update_span_name="subagent-memory-update",
            root_input={"task": task, "subagent": self.name},
            root_metadata={"agent": self.name},
        )

    def _build_core_prompt(
        self,
        task: str,
        supervisor_brief: str,
        memory_context: list[str],
    ) -> str:
        return build_subagent_core_prompt(task, supervisor_brief, memory_context)
