"""
graph_utils.py
--------------
Handles graph creation (predefined + random) and path analysis
using Dijkstra's Algorithm on risk_weight and time_cost edges.
"""

import networkx as nx
import random


# ---------------------------------------------------------------------------
# Predefined enterprise network
# ---------------------------------------------------------------------------

PREDEFINED_NODES = [
    ("UserPC_1",        {"role": "endpoint",   "sensitivity": 1}),
    ("UserPC_2",        {"role": "endpoint",   "sensitivity": 1}),
    ("PrintServer",     {"role": "server",     "sensitivity": 2}),
    ("FileServer",      {"role": "server",     "sensitivity": 3}),
    ("WebServer",       {"role": "dmz",        "sensitivity": 2}),
    ("AppServer",       {"role": "server",     "sensitivity": 3}),
    ("Database",        {"role": "database",   "sensitivity": 5}),
    ("DomainController",{"role": "critical",   "sensitivity": 5}),
    ("BackupServer",    {"role": "server",     "sensitivity": 4}),
    ("SIEM",            {"role": "security",   "sensitivity": 4}),
]

PREDEFINED_EDGES = [
    # (src, dst, risk_weight, time_cost)
    ("UserPC_1",         "FileServer",       0.15, 5),
    ("UserPC_1",         "PrintServer",      0.05, 3),
    ("UserPC_2",         "FileServer",       0.20, 6),
    ("UserPC_2",         "WebServer",        0.10, 4),
    ("PrintServer",      "FileServer",       0.08, 4),
    ("FileServer",       "AppServer",        0.25, 8),
    ("FileServer",       "BackupServer",     0.18, 7),
    ("FileServer",       "DomainController", 0.55, 12),
    ("WebServer",        "AppServer",        0.30, 7),
    ("AppServer",        "Database",         0.40, 10),
    ("AppServer",        "DomainController", 0.60, 15),
    ("Database",         "BackupServer",     0.35, 9),
    ("DomainController", "SIEM",             0.80, 20),
    ("BackupServer",     "DomainController", 0.45, 11),
    ("UserPC_1",         "UserPC_2",         0.05, 2),
]


def build_predefined_graph() -> nx.DiGraph:
    """Return the default enterprise network as a directed graph."""
    G = nx.DiGraph()

    for name, attrs in PREDEFINED_NODES:
        G.add_node(name, **attrs)

    for src, dst, risk, time in PREDEFINED_EDGES:
        G.add_edge(src, dst, risk_weight=risk, time_cost=time)
        G.add_edge(dst, src, risk_weight=risk * 0.9, time_cost=time)  # bidirectional

    return G


# ---------------------------------------------------------------------------
# Random graph generator
# ---------------------------------------------------------------------------

def build_random_graph(
    n_nodes: int = 10,
    risk_min: float = 0.05,
    risk_max: float = 0.90,
    time_min: int = 2,
    time_max: int = 20,
    edge_density: float = 0.35,
    seed: int = 42,
) -> nx.DiGraph:
    """
    Generate a random directed graph with n_nodes nodes.
    Edge density controls the probability any two nodes are connected.
    """
    random.seed(seed)
    roles = ["endpoint", "server", "database", "critical", "dmz", "security"]
    node_names = [f"Node_{i}" for i in range(n_nodes)]

    G = nx.DiGraph()
    for name in node_names:
        G.add_node(
            name,
            role=random.choice(roles),
            sensitivity=random.randint(1, 5),
        )

    # Ensure the graph is connected (random spanning tree first)
    shuffled = node_names[:]
    random.shuffle(shuffled)
    for i in range(len(shuffled) - 1):
        _add_random_edge(G, shuffled[i], shuffled[i + 1], risk_min, risk_max, time_min, time_max)

    # Add extra edges based on density
    for i, u in enumerate(node_names):
        for v in node_names:
            if u != v and not G.has_edge(u, v):
                if random.random() < edge_density:
                    _add_random_edge(G, u, v, risk_min, risk_max, time_min, time_max)

    return G


def _add_random_edge(G, u, v, risk_min, risk_max, time_min, time_max):
    risk = round(random.uniform(risk_min, risk_max), 2)
    time = random.randint(time_min, time_max)
    G.add_edge(u, v, risk_weight=risk, time_cost=time)


# ---------------------------------------------------------------------------
# Path analysis
# ---------------------------------------------------------------------------

def find_lowest_risk_path(G: nx.DiGraph, source: str, target: str):
    """
    Dijkstra's Algorithm weighted by risk_weight.
    Returns (path_list, total_risk, total_time) or (None, inf, inf).
    """
    try:
        path = nx.dijkstra_path(G, source, target, weight="risk_weight")
        total_risk = sum(
            G[path[i]][path[i + 1]]["risk_weight"] for i in range(len(path) - 1)
        )
        total_time = sum(
            G[path[i]][path[i + 1]]["time_cost"] for i in range(len(path) - 1)
        )
        return path, round(total_risk, 4), total_time
    except nx.NetworkXNoPath:
        return None, float("inf"), float("inf")
    except nx.NodeNotFound as e:
        raise ValueError(f"Node not found: {e}")


def find_shortest_time_path(G: nx.DiGraph, source: str, target: str):
    """
    Dijkstra's Algorithm weighted by time_cost.
    Returns (path_list, total_risk, total_time) or (None, inf, inf).
    """
    try:
        path = nx.dijkstra_path(G, source, target, weight="time_cost")
        total_risk = sum(
            G[path[i]][path[i + 1]]["risk_weight"] for i in range(len(path) - 1)
        )
        total_time = sum(
            G[path[i]][path[i + 1]]["time_cost"] for i in range(len(path) - 1)
        )
        return path, round(total_risk, 4), total_time
    except nx.NetworkXNoPath:
        return None, float("inf"), float("inf")
    except nx.NodeNotFound as e:
        raise ValueError(f"Node not found: {e}")


def get_edge_data_df(G: nx.DiGraph):
    """Return all edges as a list of dicts for display."""
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({
            "From": u,
            "To": v,
            "Risk Weight": data.get("risk_weight", "-"),
            "Time Cost": data.get("time_cost", "-"),
        })
    return rows


def node_risk_scores(G: nx.DiGraph) -> dict:
    """
    Compute a composite risk score per node based on
    incoming edge risk weights and node sensitivity.
    """
    scores = {}
    for node in G.nodes():
        incoming = [
            G[u][node]["risk_weight"]
            for u in G.predecessors(node)
            if "risk_weight" in G[u][node]
        ]
        avg_in = sum(incoming) / len(incoming) if incoming else 0.0
        sensitivity = G.nodes[node].get("sensitivity", 1)
        scores[node] = round(avg_in * sensitivity, 3)
    return scores