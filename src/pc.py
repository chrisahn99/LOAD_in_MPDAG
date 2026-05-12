from collections import defaultdict
from typing import Callable

import numpy as np

from src.mb_by_mb import skeleton_step, v_struc_pc, meek


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
    cpdag = meek(pdag, sep_set)

    # Convert to amat: cpdag[i,j]==-1 means tail at i, so edge exists out of i
    amat = np.zeros((n, n), dtype=int)
    amat[cpdag == -1] = 1
    return {"amat": amat.tolist()}
