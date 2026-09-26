from langgraph.graph import END, START, StateGraph

from src.core.types import MASState


def build_macnet_graph(runtime):
    workflow = StateGraph(MASState)
    workflow.add_node("supervisor", runtime.supervisor_node)
    workflow.add_node("macnet_layer", runtime.macnet_layer_node)
    workflow.add_node("macnet_decision", runtime.macnet_decision_node)
    workflow.add_node("final", runtime.final_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "macnet_layer")
    workflow.add_edge("macnet_layer", "macnet_decision")
    workflow.add_edge("macnet_decision", "final")
    workflow.add_edge("final", END)
    return workflow.compile()
