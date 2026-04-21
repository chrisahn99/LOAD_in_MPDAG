from collections import defaultdict

import numpy as np
from causallearn.utils.cit import CIT


class CountingTest:
    """
    Wrapper for CI tests that counts the number of tests done.
    """

    def __init__(
        self,
        data: np.ndarray,
        ci_test: str,
        **kwargs,
    ):
        self.cit = CIT(data, ci_test, **kwargs)
        if ci_test == "fisherz":
            self.cit.precision_matrix = np.linalg.inv(self.cit.correlation_matrix)

        self.method = self.cit.method
        self.tests_done = defaultdict(set)

    def __call__(
        self, X: int, Y: int, condition_set: list[int] | None = [], *args, **kwargs
    ) -> float:
        """
        Run the conditional independence test.

        Args:
            X (int): The first variable.
            Y (int): The second variable.
            condition_set (list[int] | None): The conditioning set.

        Returns:
            float: The p-value of the CI test.
        """
        if condition_set is None:
            condition_set = []
        self.tests_done[frozenset((X, Y))] |= {tuple(condition_set)}
        return self.cit(X, Y, condition_set)

    def get_tests_per_order(self) -> np.ndarray:
        """
        Get the number of tests done per order.

        Returns:
            np.ndarray: The number of tests done per order.
        """
        num_nodes = self.cit.data.shape[1]
        cond_sets = self.tests_done.values()
        if not cond_sets:
            return np.zeros(num_nodes, dtype=int)
        orders, test_num = np.unique(
            [len(cond) for conds in cond_sets for cond in conds],
            return_counts=True,
        )
        tests_per_order = np.zeros(num_nodes, dtype=int)
        tests_per_order[orders] = test_num
        return tests_per_order
