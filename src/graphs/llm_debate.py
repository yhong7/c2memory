from langgraph.graph import END, START, StateGraph

from src.core.types import MASState


def build_llm_debate_graph(runtime):
    workflow = StateGraph(MASState)
    workflow.add_node("supervisor", runtime.supervisor_node)
    workflow.add_node("llm_debate", runtime.llm_debate_node)
    workflow.add_node("final", runtime.final_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "llm_debate")
    workflow.add_edge("llm_debate", "final")
    workflow.add_edge("final", END)
    return workflow.compile()
