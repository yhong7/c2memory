from langgraph.graph import END, START, StateGraph

from src.core.types import MASState


def build_autogen_graph(runtime):
    workflow = StateGraph(MASState)
    workflow.add_node("supervisor", runtime.supervisor_node)
    workflow.add_node("autogen_solver", runtime.autogen_solver_node)
    workflow.add_node("autogen_recovery", runtime.autogen_recovery_node)
    workflow.add_node("final", runtime.final_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "autogen_solver")
    workflow.add_conditional_edges(
        "autogen_solver",
        runtime.autogen_router,
        {
            "recovery": "autogen_recovery",
            "final": "final",
        },
    )
    workflow.add_edge("autogen_recovery", "final")
    workflow.add_edge("final", END)
    return workflow.compile()
