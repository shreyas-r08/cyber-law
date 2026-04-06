"""
simulation.py
-------------
Models dwell time accumulation and detection probability along a traversal path.

Detection probability formula:
    P(detection) = 1 - exp(-risk_sum / (time * monitoring_strength))

A higher monitoring_strength amplifies the detection chance.
Longer dwell time (more hops, slower movement) reduces detection per unit risk
but accumulates more total exposure.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional
import networkx as nx


@dataclass
class HopMetrics:
    """Metrics for a single hop in the traversal path."""
    step: int
    from_node: str
    to_node: str
    risk_weight: float
    time_cost: int
    cumulative_risk: float
    cumulative_time: int
    hop_detection_prob: float   # detection probability for this single hop
    cumulative_detection_prob: float  # running detection probability


@dataclass
class SimulationResult:
    """Full simulation result for a path."""
    path: List[str]
    hops: List[HopMetrics] = field(default_factory=list)
    total_risk: float = 0.0
    total_time: int = 0
    final_detection_prob: float = 0.0
    monitoring_strength: float = 1.0
    label: str = "Path"


def simulate_path(
    G: nx.DiGraph,
    path: List[str],
    monitoring_strength: float = 1.0,
    label: str = "Path",
) -> Optional[SimulationResult]:
    """
    Walk the given path through G, computing per-hop and cumulative metrics.

    Parameters
    ----------
    G                  : NetworkX DiGraph
    path               : ordered list of node names
    monitoring_strength: scalar multiplier on detection sensitivity (0.1 – 3.0)
    label              : display label for the result

    Returns
    -------
    SimulationResult or None if path is invalid / too short
    """
    if path is None or len(path) < 2:
        return None

    result = SimulationResult(path=path, monitoring_strength=monitoring_strength, label=label)
    cumulative_risk = 0.0
    cumulative_time = 0

    # Probability of NOT having been detected yet (complement product)
    prob_undetected = 1.0

    for step, (u, v) in enumerate(zip(path[:-1], path[1:]), start=1):
        if not G.has_edge(u, v):
            break  # broken path, stop simulation

        edge = G[u][v]
        risk = edge.get("risk_weight", 0.0)
        time = edge.get("time_cost", 1)

        cumulative_risk += risk
        cumulative_time += time

        # Per-hop detection probability
        hop_det = _detection_prob(risk, time, monitoring_strength)

        # Update running undetected probability (independent hops)
        prob_undetected *= (1.0 - hop_det)
        cumulative_det = round(1.0 - prob_undetected, 6)

        result.hops.append(
            HopMetrics(
                step=step,
                from_node=u,
                to_node=v,
                risk_weight=round(risk, 4),
                time_cost=time,
                cumulative_risk=round(cumulative_risk, 4),
                cumulative_time=cumulative_time,
                hop_detection_prob=round(hop_det, 6),
                cumulative_detection_prob=cumulative_det,
            )
        )

    result.total_risk = round(cumulative_risk, 4)
    result.total_time = cumulative_time
    result.final_detection_prob = round(1.0 - prob_undetected, 6)
    return result


def _detection_prob(risk: float, time: int, monitoring_strength: float) -> float:
    """
    Single-hop detection probability.

        P = 1 - exp( -risk * monitoring_strength / max(time, 1) )

    Intuition:
      * Higher risk → higher probability of being flagged.
      * Longer time on hop → more opportunity to blend in → lower probability.
      * Higher monitoring_strength → security tools are more sensitive.
    """
    time = max(time, 1)
    exponent = (risk * monitoring_strength) / time
    return 1.0 - math.exp(-exponent)


def full_detection_probability(
    total_risk: float,
    total_time: int,
    monitoring_strength: float = 1.0,
) -> float:
    """
    Aggregate detection probability for the full path (simplified formula).
    Used for quick comparison display without per-hop granularity.
    """
    if total_time <= 0:
        return 1.0
    return round(1.0 - math.exp(-(total_risk * monitoring_strength) / total_time), 6)


def compare_paths(
    G: nx.DiGraph,
    risk_path: List[str],
    time_path: List[str],
    monitoring_strength: float = 1.0,
) -> dict:
    """
    Run simulation on both paths and return a comparison dict.
    """
    risk_sim = simulate_path(G, risk_path, monitoring_strength, label="Lowest-Risk Path")
    time_sim = simulate_path(G, time_path, monitoring_strength, label="Shortest-Time Path")

    return {
        "risk_path_result": risk_sim,
        "time_path_result": time_sim,
    }


def hops_to_records(sim: SimulationResult) -> List[dict]:
    """Convert HopMetrics list to plain dicts for DataFrame display."""
    return [
        {
            "Step": h.step,
            "From": h.from_node,
            "To": h.to_node,
            "Risk Weight": h.risk_weight,
            "Time Cost (units)": h.time_cost,
            "Cumulative Risk": h.cumulative_risk,
            "Cumulative Time": h.cumulative_time,
            "Hop Detection Prob": f"{h.hop_detection_prob:.2%}",
            "Cumulative Detection": f"{h.cumulative_detection_prob:.2%}",
        }
        for h in sim.hops
    ]