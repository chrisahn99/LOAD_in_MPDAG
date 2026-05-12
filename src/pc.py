from collections import defaultdict
from typing import Callable

import numpy as np

from src.mb_by_mb import skeleton_step, v_struc_pc


def _meek(pdag: np.ndarray) -> np.ndarray:
    """
    Standard Meek orientation rules (R1-R3) for CPDAGs.

    Purely structural — fires on graph topology alone with no sep_set
    conditioning, matching the original PC algorithm specification.
    All matching patterns are oriented simultaneously per rule before
    re-checking, as in the original vectorized implementation.
    """

    def rule1(pdag: np.ndarray) -> bool:
        # i -> j - k, i not adjacent to k  =>  orient j -> k
        i, j = np.where((pdag == -1) & (pdag.T == 1))  # i -> j
        k = np.where(
            ((pdag[j] == -1) & (pdag.T[j] == -1))  # j - k
            & (pdag[i] == 0)                         # i -/- k
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        pdag[k, j] = 1
        return len(i) > 0

    def rule2(pdag: np.ndarray) -> bool:
        # i -> j -> k - i  =>  orient i -> k
        i, j = np.where((pdag == -1) & (pdag.T == 1))  # i -> j
        k = np.where(
            ((pdag[j] == -1) & (pdag.T[j] == 1))    # j -> k
            & ((pdag[i] == -1) & (pdag.T[i] == -1)) # i - k
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        pdag[k, i] = 1
        return len(i) > 0

    def rule3(pdag: np.ndarray) -> bool:
        # i - j, k - i -/- j, l - i -/- j, k -> l ... wait:
        # i - k -/- j, i - l -/- j, k -> i, l -> i  =>  orient i -> j
        # (written as in orient.py: find i-j undirected, then k-i with k-/-j, etc.)
        i, j = np.where((pdag == -1) & (pdag.T == -1))  # i - j
        k = np.where(
            ((pdag[i] == -1) & (pdag.T[i] == -1))  # k - i
            & (pdag[j] == 0)                         # k -/- j
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        mask = j < k
        i, j, k = i[mask], j[mask], k[mask]
        l = np.where(
            ((pdag[i] == -1) & (pdag.T[i] == -1))  # i - l
            & ((pdag[j] == -1) & (pdag.T[j] == 1)) # j -> l
            & ((pdag[k] == -1) & (pdag.T[k] == 1)) # k -> l
        )
        i, j, k, l = i[l[0]], j[l[0]], k[l[0]], l[1]
        pdag[l, i] = 1
        return len(i) > 0

    pdag = pdag.copy()
    while rule1(pdag) or rule2(pdag) or rule3(pdag):
        continue
    return pdag


def pc(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    **kwargs,
) -> dict:
    """
    PC algorithm for global causal structure learning.

    Args:
        data (np.ndarray): The data matrix (n_samples x n_nodes).
        ci_test (Callable): CI test returning a p-value given (X, Y, conditioning_set).
        alpha (float): Significance level.

    Returns:
        dict with 'amat': CPDAG adjacency matrix as nested list.
            amat[i][j] == 1 means there is an edge with tail at i
            (i.e. i->j for a directed edge, or i-j for undirected).
            Undirected edges satisfy amat[i][j] == amat[j][i] == 1.
    """
    n = data.shape[1]
    skeleton = np.full((n, n), -1, dtype=int)
    np.fill_diagonal(skeleton, 0)
    sep_set = defaultdict(list)

    order = 0
    while np.amax(np.sum(skeleton != 0, axis=1)) > order:
        skeleton = skeleton_step(order, skeleton, ci_test, alpha, sep_set)
        order += 1

    pdag = v_struc_pc(skeleton, sep_set)
    cpdag = _meek(pdag)

    # cpdag[i,j]==-1 means tail at i (edge exists out of i toward j)
    amat = np.zeros((n, n), dtype=int)
    amat[cpdag == -1] = 1
    return {"amat": amat.tolist()}
