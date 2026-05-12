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


def _meek4(pdag: np.ndarray, sep_set: dict) -> np.ndarray:
    """
    4-rule Meek for PC with background knowledge.

    R1-R3 are the same purely structural rules as _meek.
    R4 additionally uses the sep_set from PC's global skeleton search:
    for undirected a-b, if a-c, b->c, a-d, d->c, b not adjacent to d,
    and a is in every Sep(b,d), orient a->b.

    This propagates orientations introduced by injected required edges
    that R1-R3 alone cannot deduce from graph structure.
    """

    def rule1(pdag: np.ndarray) -> bool:
        i, j = np.where((pdag == -1) & (pdag.T == 1))
        k = np.where(
            ((pdag[j] == -1) & (pdag.T[j] == -1))
            & (pdag[i] == 0)
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        pdag[k, j] = 1
        return len(i) > 0

    def rule2(pdag: np.ndarray) -> bool:
        i, j = np.where((pdag == -1) & (pdag.T == 1))
        k = np.where(
            ((pdag[j] == -1) & (pdag.T[j] == 1))
            & ((pdag[i] == -1) & (pdag.T[i] == -1))
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        pdag[k, i] = 1
        return len(i) > 0

    def rule3(pdag: np.ndarray) -> bool:
        i, j = np.where((pdag == -1) & (pdag.T == -1))
        k = np.where(
            ((pdag[i] == -1) & (pdag.T[i] == -1))
            & (pdag[j] == 0)
        )
        i, j, k = i[k[0]], j[k[0]], k[1]
        mask = j < k
        i, j, k = i[mask], j[mask], k[mask]
        l = np.where(
            ((pdag[i] == -1) & (pdag.T[i] == -1))
            & ((pdag[j] == -1) & (pdag.T[j] == 1))
            & ((pdag[k] == -1) & (pdag.T[k] == 1))
        )
        i, j, k, l = i[l[0]], j[l[0]], k[l[0]], l[1]
        pdag[l, i] = 1
        return len(i) > 0

    def rule4(pdag: np.ndarray, sep_set: dict) -> bool:
        # For undirected a-b: if a-c (undirected), b->c (directed),
        # a-d (undirected), d->c (directed), b not adjacent to d,
        # and a in every Sep(b,d) => orient a->b.
        # Uses sep_set from skeleton search to certify a is non-collider
        # between b and d, making a->b the only orientation that avoids
        # a new unshielded collider.
        a, b = np.where((pdag == -1) & (pdag.T == -1))
        changed = False
        for a_, b_ in zip(a, b):
            c_list = np.where(
                ((pdag[a_] == -1) & (pdag.T[a_] == -1))   # a - c
                & ((pdag[b_] == -1) & (pdag.T[b_] == 1))  # b -> c
            )[0]
            for c_ in c_list:
                d_list = np.where(
                    ((pdag[a_] == -1) & (pdag.T[a_] == -1))   # a - d
                    & ((pdag[c_] == -1) & (pdag.T[c_] == 1))  # d -> c
                    & (pdag[b_] == 0)                           # b -/- d
                )[0]
                for d_ in d_list:
                    if frozenset((b_, d_)) in sep_set and all(
                        a_ in S for S in sep_set[frozenset((b_, d_))]
                    ):
                        pdag[b_, a_] = 1  # a -> b
                        changed = True
                        break
                if changed:
                    break
        return changed

    pdag = pdag.copy()
    while rule1(pdag) or rule2(pdag) or rule3(pdag) or rule4(pdag, sep_set):
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


def pc_with_bk(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    background_knowledge: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """
    PC algorithm with required background-knowledge edges injected after the
    initial CPDAG is built, then propagated with 4-rule Meek.

    Pipeline:
      1. Standard PC: skeleton search + v-structure orientation + 3-rule Meek
      2. Overlay required edges from background_knowledge onto the CPDAG
      3. 4-rule Meek (R1-R3 structural + R4 with sep_set) to propagate BK

    Forbidden edges in background_knowledge are ignored; only required edges
    (non-zero entries in the matrix) are injected.

    Args:
        data (np.ndarray): Data matrix (n_samples x n_nodes).
        ci_test (Callable): CI test returning a p-value.
        alpha (float): Significance level.
        background_knowledge (np.ndarray | None): Adjacency matrix where
            bk[i,j]==-1 and bk[j,i]==1 encodes the required edge i->j.
            If None, behaves identically to pc().

    Returns:
        dict with 'amat': CPDAG adjacency matrix as nested list (same
        convention as pc()).
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

    # First Meek sweep: 3 structural rules, standard PC
    cpdag = _meek(pdag)

    # Inject required edges from background knowledge
    if background_knowledge is not None:
        mask = background_knowledge != 0
        cpdag[mask] = background_knowledge[mask]

    # Second Meek sweep: all 4 rules, R4 uses sep_set to propagate BK
    cpdag = _meek4(cpdag, sep_set)

    amat = np.zeros((n, n), dtype=int)
    amat[cpdag == -1] = 1
    return {"amat": amat.tolist()}
