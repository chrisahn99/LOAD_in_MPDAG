# Modify sample_background_knowledge so that it takes two arguments in account,
# fraction_required and fraction_forbidden

import random
import math

import numpy as np
import networkx as nx


def sample_background_knowledge_v2(
      true_dag: nx.DiGraph,
      fraction_required: float = 0.1,
      fraction_forbidden: float = 0.1,
      seed: int = 42
    ):
    """
    Samples required and forbidden edges from the ground truth DAG.
    """
    random.seed(seed)
    nodes = list(true_dag.nodes())
    all_possible_edges = [(u, v) for u in nodes for v in nodes if u != v]
    true_edges = set(true_dag.edges())

    # 1. Required: Edges that actually exist in the true DAG
    num_required = int(len(true_edges) * fraction_required)
    required = set(random.sample(list(true_edges), num_required))

    # 2. Forbidden: Edges that do NOT exist in the true DAG
    # (Including the reverse of true edges to make it challenging)
    non_edges = [edge for edge in all_possible_edges if edge not in true_edges]
    num_forbidden = int(len(non_edges) * fraction_forbidden)
    forbidden = set(random.sample(non_edges, num_forbidden))

    # print(f"Required: {len(required)}, Forbidden: {len(forbidden)}")

    return {"forbidden": forbidden, "required": required}


def sample_local_background_knowledge(
    true_dag: nx.DiGraph,
    targets: np.ndarray,
    fraction_required: float = 0.1,
    fraction_forbidden: float = 0.1,
    seed: int = 42
):
    """
    Samples required and forbidden edges specifically connected to target nodes.
    """
    random.seed(seed)
    nodes = list(true_dag.nodes())
    target_set = set(targets)
    true_edges = list(true_dag.edges())

    # 1. Identify "Local" True Edges (pointing to or from targets)
    local_true_edges = [
        (u, v) for u, v in true_edges
        if u in target_set or v in target_set
    ]

    # 2. Identify "Local" Non-Edges
    # Every possible pair involving at least one target minus existing edges
    all_possible_local = [
        (u, v) for u in nodes for v in nodes
        if u != v and (u in target_set or v in target_set)
    ]
    local_non_edges = [e for e in all_possible_local if e not in set(true_edges)]

    # Helper to calculate count with a minimum of 1 if fraction > 0
    def get_sample_count(pool, fraction):
        if fraction <= 0 or not pool:
            return 0
        return max(1, math.ceil(len(pool) * fraction))

    # Sampling
    num_req = get_sample_count(local_true_edges, fraction_required)
    num_forb = get_sample_count(local_non_edges, fraction_forbidden)

    required = set(random.sample(local_true_edges, num_req)) if num_req > 0 else set()
    forbidden = set(random.sample(local_non_edges, num_forb)) if num_forb > 0 else set()

    return {"forbidden": forbidden, "required": required}


def initialize_background_knowledge(num_nodes: int, bk_dict: dict) -> np.ndarray:
    """
    Adapts a dictionary of background knowledge into an initial MPDAG adjacency matrix.

    Args:
        num_nodes (int): Total number of variables/nodes in the dataset.
        bk_dict (dict): Dictionary containing 'required' and 'forbidden' edge sets.
                        Example: {"forbidden": {(0, 1)}, "required": {(2, 3)}}
                        where a tuple (i, j) implies the directed edge i -> j.

    Returns:
        np.ndarray: The initialized graph matrix G.
    """
    # Initialize an empty graph (0 means no edge)
    g = np.zeros((num_nodes, num_nodes), dtype=int)

    # 1. Process Required Edges
    # A required edge (i, j) means i -> j is known to exist.
    if "required" in bk_dict:
        for i, j in bk_dict["required"]:
            g[i, j] = -1  # Tail at i
            g[j, i] = 1   # Arrowhead at j

    # 2. Process Forbidden Edges
    # Note: Since the initialization matrix starts with all 0s (no edges),
    # forbidden edges are technically naturally excluded from the initial G.
    # However, to strictly enforce forbidden edges in the *entire* algorithm,
    # you would normally also pass 'forbidden' into `learn_local_structure`
    # to prevent it from drawing those edges during the skeleton/Meek phases.

    return g