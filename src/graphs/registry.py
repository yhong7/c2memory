from src.graphs.autogen import build_autogen_graph
from src.graphs.dylan import build_dylan_graph
from src.graphs.llm_debate import build_llm_debate_graph
from src.graphs.macnet import build_macnet_graph
from src.graphs.supervisor_fanout import build_supervisor_fanout_graph


GRAPH_BUILDERS = {
    "supervisor_fanout": build_supervisor_fanout_graph,
    "dylan": build_dylan_graph,
    "llm_debate": build_llm_debate_graph,
    "autogen": build_autogen_graph,
    "macnet": build_macnet_graph,
}


def build_graph(graph_name: str, runtime):
    if graph_name not in GRAPH_BUILDERS:
        raise ValueError(f"Unsupported graph topology: {graph_name}")
    return GRAPH_BUILDERS[graph_name](runtime)
