"""DyLAN (Dynamic Layer-Agent Network) implemented as a LangGraph StateGraph.

Architecture
------------
START -> supervisor -> dylan_layer (reusable, loops up to max_layers)
                            ↑                   ↓ consensus? or current_layer >= max?
                            └── "next_layer"    ↓
                                          dylan_decision
                                                 ↓
                                           dylan_final
                                                 ↓
                                               END

A single *dylan_layer* node runs all active agents in parallel
(via ``asyncio.gather()``). After execution the router checks for:

1. **Consensus** — ⅔ majority among active agents → route to
   ``dylan_decision`` immediately (early exit).
2. **Max layers** — ``current_layer >= max_layers`` → route to
   ``dylan_decision`` (forced exit).
3. **Midpoint ranking** — at iteration ⌊(max_layers-1)/2⌋ the ranker agent
   evaluates all active agents and deactivates the worst one
   (pruning), shrinking the active set for remaining iterations.

Otherwise the router loops back to ``dylan_layer`` for another iteration.
"""

from langgraph.graph import END, START, StateGraph

from src.core.types import DyLANState


def build_dylan_graph(runtime):
    """Construct the DyLAN LangGraph StateGraph with a reusable loop node.

    Parameters
    ----------
    runtime : MinimalLangGraphRuntime
        Must have ``graph_name == "dylan"`` so that all DyLAN
        fields / node factories are available.
    """
    workflow = StateGraph(DyLANState)

    # ── nodes ────────────────────────────────────────────────────
    workflow.add_node("supervisor", runtime.supervisor_node)
    workflow.add_node("dylan_layer", runtime.make_dylan_layer_node())
    workflow.add_node("dylan_decision", runtime.dylan_decision_node)
    workflow.add_node("dylan_final", runtime.dylan_final_node)

    # ── edges ────────────────────────────────────────────────────
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "dylan_layer")

    # dylan_layer loops back to itself until consensus or max_layers
    workflow.add_conditional_edges(
        "dylan_layer",
        runtime.dylan_router,
        {
            "next_layer": "dylan_layer",
            "final_decision": "dylan_decision",
        },
    )

    workflow.add_edge("dylan_decision", "dylan_final")
    workflow.add_edge("dylan_final", END)

    return workflow.compile()
