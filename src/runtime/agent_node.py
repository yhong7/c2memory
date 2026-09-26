from dataclasses import dataclass
from typing import Any, Callable

from src.agents import BaseAgent
from src.api import observe
from src.memory import MemoryRetrieverBase, MemoryUpdaterBase


PromptBuilder = Callable[[list[str]], str]


@dataclass
class AgentNodeBase:
    name: str
    core_agent: BaseAgent
    retriever: MemoryRetrieverBase
    updater: MemoryUpdaterBase

    async def run_node(
        self,
        *,
        task: str,
        prompt_builder: PromptBuilder,
        root_span_name: str,
        retrieve_span_name: str,
        execute_span_name: str,
        update_span_name: str,
        trace_context: dict[str, str] | None = None,
        root_input: dict[str, Any] | None = None,
        root_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[str], str]:
        with observe(
            root_span_name,
            input=root_input or {"task": task, "agent": self.name},
            metadata=root_metadata or {"agent": self.name},
            trace_context=trace_context,
        ) as root_obs:
            with observe(
                retrieve_span_name,
                input={"task": task, "agent": self.name},
                metadata={"agent": self.name},
            ) as retrieve_obs:
                memory_context = self.retriever.retrieve(self.name, task)
                if retrieve_obs is not None:
                    retrieve_obs.update(output={"memory_context": memory_context})

            prompt = prompt_builder(memory_context)
            with observe(
                execute_span_name,
                input={"task": task, "agent": self.name, "prompt": prompt},
                metadata={"agent": self.name},
            ) as exec_obs:
                output = await self.core_agent.respond(prompt)
                if exec_obs is not None:
                    exec_obs.update(output={"response": output})

            with observe(
                update_span_name,
                input={"task": task, "agent": self.name, "output": output},
                metadata={"agent": self.name},
            ) as update_obs:
                self.updater.update(self.name, task, output)
                if update_obs is not None:
                    update_obs.update(output={"updated": True})

            if root_obs is not None:
                root_obs.update(output={"memory_context": memory_context, "output": output})
            return memory_context, output
