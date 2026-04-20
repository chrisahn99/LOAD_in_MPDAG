from collections import defaultdict
from itertools import chain, combinations
from typing import Callable, NamedTuple, Sequence

import numpy as np

from src.mb_by_mb import mb_by_mb_alg, mb_by_mb_in_mpdag_alg

## Original

class Neighbors(NamedTuple):
    """
    Represents the different types of neighbors for a node in a causal graph.
    """

    parents: set[int]
    children: set[int]
    unoriented: set[int]


def draw_complex_graph(array):
    arr = np.array(array)
    G_directed = nx.DiGraph()
    G_undirected = nx.Graph()

    nodes = range(len(arr))
    G_directed.add_nodes_from(nodes)
    G_undirected.add_nodes_from(nodes)

    for i in range(len(arr)):
        for j in range(len(arr)):
            # Handle the arrows (Heads)
            if arr[i, j] == 1:
                G_directed.add_edge(j, i)

            # Handle the links (Tails)
            # We check i < j to avoid adding the same undirected edge twice
            elif arr[i, j] == -1 and arr[j, i] == -1 and i < j:
                G_undirected.add_edge(i, j)

    pos = nx.circular_layout(G_directed)
    plt.figure(figsize=(8, 6))

    # 1. Draw Nodes
    nx.draw_networkx_nodes(G_directed, pos, node_color='skyblue')
    nx.draw_networkx_labels(G_directed, pos)

    # 2. Draw Directed Edges (Arrows)
    nx.draw_networkx_edges(G_directed, pos, edgelist=G_directed.edges(), arrows=True)

    # 3. Draw Undirected Edges (Simple Lines)
    nx.draw_networkx_edges(G_undirected, pos, edgelist=G_undirected.edges(), arrows=False)

    plt.show()


def get_neighbors(g: np.ndarray, x: int) -> Neighbors:
    """
    Get the neighbors of a target node in the graph.

    Args:
        g (np.ndarray): The local graph of x.
        x (int): The target node.

    Returns:
        Neighbors: The parents, children, and unoriented neighbors of the target node.
    """
    parents = set(np.where((g[x] == 1) & (g[:, x] == -1))[0])
    children = set(np.where((g[x] == -1) & (g[:, x] == 1))[0])
    unoriented = set(np.where((g[x] == -1) & (g[:, x] == -1))[0])
    return Neighbors(parents, children, unoriented)


def is_explicit_ancestor(
    x: int,
    y: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Check if x is an explicit ancestor of y.

    Args:
        x (int): The node to check as ancestor.
        y (int): The node to check as descendant.
        g (np.ndarray): The local graph of x.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.

    Returns:
        bool: True if x is an explicit ancestor of y, False otherwise.
    """
    neighbors = get_neighbors(g, x)
    # print(f"Neighbors of node {x} are: {neighbors}")
    if y in neighbors.children:
        # print("direct connection")
        return True
    elif y in neighbors.parents | neighbors.unoriented:
        return False

    # print("performing ci test")

    # print(f"Value of ci test for {x} an ancestor of {y}: {ci_test(x, y, neighbors.parents | neighbors.unoriented)}")
    return ci_test(x, y, neighbors.parents | neighbors.unoriented) < alpha


def is_possible_ancestor(
    x: int,
    y: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Check if x is a possible ancestor of y.

    Args:
        x (int): The node to check as ancestor.
        y (int): The node to check as descendant.
        g (np.ndarray): The local graph of x.
        ci_test (Callable): Conditional independence test.
        alpha (float): Significance level.

    Returns:
        bool: True if x is a possible ancestor of y, False otherwise.
    """
    neighbors = get_neighbors(g, x)
    if y in neighbors.children | neighbors.unoriented:
        return True
    elif y in neighbors.parents:
        return False
    return ci_test(x, y, neighbors.parents) < alpha


def is_amenable(
    x: int,
    y: int,
    v: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Locally test if an undirected neighbor of the treatment has no undirected paths
    to the outcome that do not go through the treatment.

    Args:
        x (int): The treatment node.
        y (int): The outcome node.
        v (int): An undirected neighbor of the treatment.
        g (np.ndarray): The local graph of v.
        ci_test (Callable): Conditional independence test.
        alpha (float): Significance level.

    Returns:
        bool: True if v does not contradict amenability, otherwise False.
    """
    neighbors = get_neighbors(g, v)
    if y in neighbors.children | neighbors.unoriented | neighbors.parents:  # adjacent
        return False
    return ci_test(v, y, neighbors.parents | {x}) >= alpha


def get_locally_valid_parent_sets(g: np.ndarray, x: int) -> Sequence[set[int]]:
    """
    Get all locally valid parent sets for a given node in the graph, following
    the local IDA algorithm (Algorithm 3) in
    Estimating high-dimensional intervention effects from observational data
    by Marloes H. Maathuis, Markus Kalisch and Peter Bühlmann

    Args:
        g (np.ndarray): The local graph of x.
        x (int): The target node.

    Returns:
        Sequence[set[int]]: The locally valid parent sets.
    """
    neighbors = get_neighbors(g, x)
    skeleton = g != 0
    np.fill_diagonal(skeleton, True)

    valid_sets = []
    # for each subset of unoriented neighbors of x
    for new_parents in chain.from_iterable(
        combinations(neighbors.unoriented, r)
        for r in range(len(neighbors.unoriented) + 1)
    ):
        candidate_set = neighbors.parents.union(new_parents)
        # Check if the candidate set is locally valid, i.e., has no NEW v-structure
        # by checking if all NEW parents are neighbours of all current parents
        if np.all(skeleton[new_parents, :][:, list(candidate_set)]):
            valid_sets.append(candidate_set)
    return valid_sets


def load(
    data: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    targets: Sequence[int],
    mb_algorithm: str = "grow_shrink",
    ignore: Sequence[int] = [],
    logging: bool = False,
    **kwargs,
) -> dict:
    """
    Optimal adjustment set discovery using local causal discovery algorithms.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (Sequence[int]): The target nodes.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        ignore (Sequence[int]): Nodes to ignore in discovery process.
        **kwargs: Additional keyword arguments are ignored.
    Returns:
        dict: Adjustment sets and a boolean indicating if the causal effect is identifiable.
    """
    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)

    adj_sets = dict()

    t1, t2 = targets
    if logging:
      print("========================")
      print(f"Running mb by mb for {t1}")
    G[t1] = mb_by_mb_alg(
        data, ci_test, alpha, t1, mb_algorithm, MB, L, sep_set, ignore=ignore
    )
    if logging:
      print("========================")
      print(f"After applications of Meek's rules on {t1} we have:")
      draw_complex_graph(G[t1])


    if logging:
      print("========================")
      print(f"Running mb by mb for {t2}")
    G[t2] = mb_by_mb_alg(
        data, ci_test, alpha, t2, mb_algorithm, MB, L, sep_set, ignore=ignore
    )
    if logging:
      print("========================")
      print(f"After applications of Meek's rules on {t2} we have:")
      draw_complex_graph(G[t2])

    # Determine causal relationship
    if is_explicit_ancestor(t1, t2, G[t1], ci_test, alpha):
        treatment, outcome = t1, t2
    elif is_explicit_ancestor(t2, t1, G[t2], ci_test, alpha):
        treatment, outcome = t2, t1
    else:
        if is_possible_ancestor(t1, t2, G[t1], ci_test, alpha):
            adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1)
        if is_possible_ancestor(t2, t1, G[t2], ci_test, alpha):
            adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2)
        return {
            "adj_sets": str(adj_sets),
            "identifiable": False,
            "id_tests": ci_test.get_tests_per_order().tolist(),
        }

    # Check if amenable
    unoriented = get_neighbors(G[treatment], treatment).unoriented
    for v in unoriented:
        G[v] = mb_by_mb_alg(data, ci_test, alpha, v, mb_algorithm, MB, L, sep_set)
        if not is_amenable(treatment, outcome, v, G[v], ci_test, alpha):
            adj_sets[(treatment, outcome)] = get_locally_valid_parent_sets(
                G[treatment], treatment
            )
            return {
                "adj_sets": str(adj_sets),
                "identifiable": False,
                "id_tests": ci_test.get_tests_per_order().tolist(),
            }
    id_tests = ci_test.get_tests_per_order().tolist()

    # Identify explicit descendants of treatment
    desc = set()
    for v in set(range(data.shape[1])) - {treatment, outcome}:
        if is_explicit_ancestor(treatment, v, G[treatment], ci_test, alpha):

            if logging:
              print("========================")
              print(f"Adding {v} to desc")

            desc.add(v)
            G[v] = mb_by_mb_alg(data, ci_test, alpha, v, mb_algorithm, MB, L, sep_set)

    # Identify explicit mediators between treatment and outcome
    #print("mediators")
    meds = set()
    for v in desc:
        if is_explicit_ancestor(v, outcome, G[v], ci_test, alpha):
            meds.add(v)
            if logging:
              print("========================")
              print(f"Adding {v} to oset and drawing local graph")
              draw_complex_graph(G[v])

    # Identify optimal adjustment set
    #print("identiying oset")
    oset = set()
    for med in meds | {outcome}:
        oset |= get_neighbors(G[med], med).parents
        if logging:
          print("========================")
          print(f"Parent nodes of mediator {med}: {get_neighbors(G[med], med).parents}")

    oset -= meds | {treatment}
    if logging:
      print("========================")
      print(f"Final oset: {oset}")

    adj_sets[(treatment, outcome)] = [oset]
    return {"adj_sets": str(adj_sets), "identifiable": True, "id_tests": id_tests}


## LOAD in MPDAG version

def load_in_mpdag(
    data: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    targets: Sequence[int],
    background_knowledge: np.ndarray | None = None,
    mb_algorithm: str = "grow_shrink",
    ignore: Sequence[int] = [],
    logging: bool = False,
    **kwargs,
) -> dict:
    """
    Optimal adjustment set discovery using local causal discovery algorithms.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (Sequence[int]): The target nodes.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        ignore (Sequence[int]): Nodes to ignore in discovery process.
        **kwargs: Additional keyword arguments are ignored.
    Returns:
        dict: Adjustment sets and a boolean indicating if the causal effect is identifiable.
    """
    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)

    adj_sets = dict()

    t1, t2 = targets
    if logging:
      print("========================")
      print(f"Running mb by mb for {t1}")
    G[t1] = mb_by_mb_in_mpdag_alg(
        data, ci_test, alpha, t1, background_knowledge, mb_algorithm, MB, L, sep_set, ignore=ignore
    )
    if logging:
      print("========================")
      print(f"After applications of Meek's rules on {t1} we have:")
      draw_complex_graph(G[t1])


    if logging:
      print("========================")
      print(f"Running mb by mb for {t2}")
    G[t2] = mb_by_mb_in_mpdag_alg(
        data, ci_test, alpha, t2, background_knowledge, mb_algorithm, MB, L, sep_set, ignore=ignore
    )
    if logging:
      print("========================")
      print(f"After applications of Meek's rules on {t2} we have:")
      draw_complex_graph(G[t2])

    # Determine causal relationship
    if is_explicit_ancestor(t1, t2, G[t1], ci_test, alpha):
        treatment, outcome = t1, t2
    elif is_explicit_ancestor(t2, t1, G[t2], ci_test, alpha):
        treatment, outcome = t2, t1
    else:
        if is_possible_ancestor(t1, t2, G[t1], ci_test, alpha):
            adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1)
        if is_possible_ancestor(t2, t1, G[t2], ci_test, alpha):
            adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2)
        return {
            "adj_sets": str(adj_sets),
            "identifiable": False,
            "id_tests": ci_test.get_tests_per_order().tolist(),
        }

    # Check if amenable
    unoriented = get_neighbors(G[treatment], treatment).unoriented
    for v in unoriented:
        G[v] = mb_by_mb_in_mpdag_alg(data, ci_test, alpha, v, background_knowledge, mb_algorithm, MB, L, sep_set)
        if not is_amenable(treatment, outcome, v, G[v], ci_test, alpha):
            adj_sets[(treatment, outcome)] = get_locally_valid_parent_sets(
                G[treatment], treatment
            )
            return {
                "adj_sets": str(adj_sets),
                "identifiable": False,
                "id_tests": ci_test.get_tests_per_order().tolist(),
            }
    id_tests = ci_test.get_tests_per_order().tolist()

    # Identify explicit descendants of treatment
    desc = set()
    for v in set(range(data.shape[1])) - {treatment, outcome}:
        if is_explicit_ancestor(treatment, v, G[treatment], ci_test, alpha):

            if logging:
              print("========================")
              print(f"Adding {v} to desc")

            desc.add(v)
            G[v] = mb_by_mb_in_mpdag_alg(data, ci_test, alpha, v, background_knowledge, mb_algorithm, MB, L, sep_set)

    # Identify explicit mediators between treatment and outcome
    #print("mediators")
    meds = set()
    for v in desc:
        if is_explicit_ancestor(v, outcome, G[v], ci_test, alpha):
            meds.add(v)
            if logging:
              print("========================")
              print(f"Adding {v} to oset and drawing local graph")
              draw_complex_graph(G[v])

    # Identify optimal adjustment set
    #print("identiying oset")
    oset = set()
    for med in meds | {outcome}:
        oset |= get_neighbors(G[med], med).parents
        if logging:
          print("========================")
          print(f"Parent nodes of mediator {med}: {get_neighbors(G[med], med).parents}")

    oset -= meds | {treatment}
    if logging:
      print("========================")
      print(f"Final oset: {oset}")

    adj_sets[(treatment, outcome)] = [oset]
    return {"adj_sets": str(adj_sets), "identifiable": True, "id_tests": id_tests}