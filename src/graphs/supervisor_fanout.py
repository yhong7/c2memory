from langgraph.graph import END, START, StateGraph

from src.core.types import MASState


def build_supervisor_fanout_graph(runtime):
    workflow = StateGraph(MASState)
    workflow.add_node("supervisor", runtime.supervisor_node)

    for subagent_name in runtime.subagent_names:
        workflow.add_node(subagent_name, runtime.make_subagent_node(subagent_name))

    workflow.add_node("final", runtime.final_node)

    workflow.add_edge(START, "supervisor")
    for subagent_name in runtime.subagent_names:
        workflow.add_edge("supervisor", subagent_name)
        workflow.add_edge(subagent_name, "final")
    workflow.add_edge("final", END)
    return workflow.compile()
