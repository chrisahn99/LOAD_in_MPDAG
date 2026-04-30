import os
from pathlib import Path
from itertools import permutations

import numpy as np
import networkx as nx
from tqdm import tqdm
from rpy2.robjects import default_converter, globalenv, numpy2ri, r
from rpy2 import robjects

# Source the R evaluation functions from the project root
file_path = Path(__file__).resolve().parent.parent / "evaluate.R"

# Source the file using the R 'source' function via robjects
if file_path.exists():
    robjects.r.source(str(file_path))
    print(f"Successfully sourced: {file_path}")
else:
    print(f"Error: File not found at {file_path}")
    raise FileNotFoundError(f"R evaluation script not found at {file_path}")


def dag2cpdag(dag: np.ndarray) -> np.ndarray:
    with (default_converter + numpy2ri.converter).context():
        cpdag = r["dag2cpdag"](r["as"](dag != 0, "graphNEL"))
        cpdag = np.array(r["as"](cpdag, "matrix")).astype(np.int8)
    return cpdag


# TESTS & TIME
def get_test_nums(results):
    test_sum = [np.sum(h["tests"]) for h in results]
    return np.array(test_sum)


def get_times(results):
    return np.array([h["time"] for h in results])


# ADJ-SETS
def get_optimal_adj_set(amat, treatment, outcome):
    if not is_ancestor(treatment, outcome, amat):
        return None
    try:
        with (default_converter + numpy2ri.converter).context():
            S = r["optAdjSet"](
                (r["as"](amat != 0, "graphNEL")), treatment + 1, outcome + 1
            )
        return set(np.array(S) - 1)
    except Exception:
        return None


def get_true_osets(experiments):
    """
    Sequentially determines the true optimal adjustment sets for a dictionary of experiments.
    """
    true_osets = {}

    for exp_id, exp in experiments.items():
        # 1. Convert the true DAG to a CPDAG
        # Using a helper (dag2cpdag) to represent the Markov Equivalence Class
        true_dag_matrix = nx.to_numpy_array(exp["true_dag"])
        cpdag = dag2cpdag(true_dag_matrix)

        # 2. Identify treatment and outcome from the true DAG
        # Based on explicit ancestral relations
        targets = exp["targets"]
        if nx.has_path(exp["true_dag"], targets[0], targets[1]):
            treatment, outcome = targets[0], targets[1]
        else:
            treatment, outcome = targets[1], targets[0]

        # 3. Retrieve the true optimal adjustment set relative to the CPDAG
        true_oset = get_optimal_adj_set(cpdag, treatment, outcome)

        # 4. Store the ground truth result
        true_osets[exp_id] = {
            "treatment": treatment,
            "outcome": outcome,
            "oset": true_oset,
        }

        # print(f"Processed Experiment {exp_id}: Treatment={treatment}, Outcome={outcome}")

    return true_osets

# EVALUATE OSET
def get_precision(true: set, pred: set):
    if true == pred:
        return 1.0
    tp = len(true.intersection(pred))
    fp = len(pred - true)
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def get_recall(true: set, pred: set):
    if true == pred:
        return 1.0
    tp = len(true.intersection(pred))
    fn = len(true - pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def get_f1(precision: float, recall: float):
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


def get_oset(project: str, h: dict, treatment: int, outcome: int):
    if project in ["pc", "fges", "marvel", "snap"]:
        est_oset = get_optimal_adj_set(np.array(h["amat"]), treatment, outcome)
        if est_oset is not None:
            est_osets = [est_oset]
        else:
            est_osets = None
    elif project in ["mb_by_mb", "ldecc", "mb_by_mb_plus", "ldecc_plus"]:
        adj_sets = eval(h["adj_sets"])
        if (treatment, outcome) in adj_sets:
            est_osets = adj_sets[(treatment, outcome)]
        else:
            est_osets = None
    elif project in ["ldp", "ldp_plus"]:
        results = eval(h["results"])
        if (treatment, outcome) in results and results[(treatment, outcome)][
            "vas_exists"
        ]:
            parts = results[(treatment, outcome)]
            est_osets = [
                set(parts["Z1"]),
                set(parts["Z1"] + parts["Z4"]),
                set(parts["Z1"] + parts["Z4"] + parts["Z5"]),
            ]
        else:
            est_osets = None
    elif project in ["load", "load_oracle"]:
        adj_sets = eval(h["adj_sets"])
        if h["identifiable"] and (treatment, outcome) in adj_sets:
            est_osets = adj_sets[(treatment, outcome)]
        else:
            est_osets = None
    else:
        raise ValueError("Invalid project")
    return est_osets


def evaluate_oset(algorithm, results, true_osets: dict):
    """
    Sequentially evaluates discovered adjustment sets against ground truth.
    Returns trimmed arrays (removing outliers) for precision, recall, and F1.
    """
    prec_scores = []
    rec_scores = []
    f1_scores = []

    for h in results:
        # 1. Handle algorithm failures
        if "failed" in h and h["failed"]:
            prec_scores.append(0.0)
            rec_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        exp_id = h["id"]
        treatment = true_osets[exp_id]["treatment"]
        outcome = true_osets[exp_id]["outcome"]

        # 2. Get estimated optimal adjustment sets
        # In a sequential run, we call the function directly
        est_osets = get_oset(algorithm, h, treatment, outcome)
        true_oset = true_osets[exp_id]["oset"]

        # 3. Calculate Scores (Precision, Recall, F1)
        # Based on the paper's metrics for adjustment set quality
        if true_oset is not None and est_osets is not None:
            precs = [get_precision(true_oset, s) for s in est_osets]
            recs = [get_recall(true_oset, s) for s in est_osets]
            f1s = [get_f1(p, r) for p, r in zip(precs, recs)]

            # Record the best match among returned sets
            prec_scores.append(max(precs))
            rec_scores.append(max(recs))
            f1_scores.append(max(f1s))

        elif true_oset == est_osets:  # Both are None (case for non-existent Oset)
            prec_scores.append(1.0)
            rec_scores.append(1.0)
            f1_scores.append(1.0)
        else:
            # Case where one exists and the other doesn't
            prec_scores.append(0.0)
            rec_scores.append(0.0)
            f1_scores.append(0.0)

    # 4. Return results
    return (
        (prec_scores),
        (rec_scores),
        (f1_scores),
    )


# INTERVENTION DISTANCE
def is_ancestor(t1: int, t2: int, amat: np.ndarray) -> bool:
    reach = amat.copy().astype(bool)
    np.fill_diagonal(reach, True)
    reach = np.linalg.matrix_power(reach, reach.shape[0] - 1)
    return reach[t1, t2]


def true_linear_gaussian_effect(
    treatment: int, outcome: int, true_dag: nx.DiGraph, **kwargs
) -> float:
    if treatment not in nx.ancestors(true_dag, outcome):
        return 0.0
    amat = nx.to_numpy_array(true_dag)
    return sum(
        np.prod([amat[path[i], path[i + 1]] for i in range(len(path) - 1)])
        for path in nx.all_simple_paths(true_dag, treatment, outcome)
    )


def true_binary_effect(
    treatment: int, outcome: int, true_dag: nx.DiGraph, cpt: object, **kwargs
) -> float:
    if treatment not in nx.ancestors(true_dag, outcome):
        return 0.0
    with (default_converter + numpy2ri.converter).context():
        return globalenv["true_binary_effect"](
            int(treatment) + 1, int(outcome) + 1, nx.to_numpy_array(true_dag), cpt
        )[0]


def true_causal_effects(experiments: dict, family="gaussian") -> dict:
    if family == "gaussian":
        get_effect = true_linear_gaussian_effect
    elif family == "binary":
        get_effect = true_binary_effect
    else:
        raise ValueError("Invalid family")
    effects = {}
    for exp_id, exp in tqdm(experiments.items(), desc="Getting true effects"):
        effects[exp_id] = {}
        for t1, t2 in permutations(exp["targets"], 2):
            effects[exp_id][(t1, t2)] = get_effect(
                treatment=t1, outcome=t2, true_dag=exp["true_dag"], cpt=exp["cpt"]
            )
    return effects


def estimate_binary(samples, val, outcome, treatment, adj_set):
    # Filter rows based on the treatment value
    samples_val = samples[samples[:, treatment] == val]
    # Create the design matrix X
    X = np.column_stack(
        [np.ones(len(samples_val))] + [samples_val[:, idx] for idx in adj_set]
    )
    # Outcome variable (filtered by val)
    y = samples_val[:, outcome]
    # Solve the least squares problem X * beta = y
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # Predict the outcome for all rows in the original samples using the estimated beta
    X_full = np.column_stack(
        [np.ones(len(samples))] + [samples[:, idx] for idx in adj_set]
    )
    y_pred = X_full @ beta
    return y_pred.mean()


def estimate_ate(
    treatment: int,
    outcome: int,
    adj_sets: list[list[int]],
    samples: np.ndarray,
    family: str = "gaussian",
) -> list[float]:
    effects = []
    for adj_set in adj_sets:
        adj_set = list(adj_set)
        if family == "gaussian":
            x = samples[:, [treatment] + adj_set]
            y = samples[:, outcome]
            A = np.vstack((x.T, np.ones(len(x))))
            effect = np.linalg.lstsq(A.T, y, rcond=None)[0][0]
            effects.append(effect)
        elif family == "binary":
            do_1 = estimate_binary(samples, 1, outcome, treatment, adj_set)
            do_0 = estimate_binary(samples, 0, outcome, treatment, adj_set)
            effects.append(do_1 - do_0)
    return effects


def get_adj_sets(project: str, h: dict, t1: int, t2: int):
    if "failed" in h and h["failed"]:
        return [[]]
    if project in ["pc", "fges", "marvel", "snap"]:
        amat = np.array(h["amat"], dtype=np.int8)
        if is_ancestor(t1, t2, amat):
            oset = get_optimal_adj_set(amat, t1, t2)
            if oset != None:  # identifiable
                return [oset]
            else:  # unidentifiable, fall back to local IDA
                amat = -amat.copy()
                amat[np.logical_and(amat == 0, amat.T == -1)] = 1
                # return get_locally_valid_parent_sets(amat, t1, t2)
                return [[]] # Placeholder as we don't have IDA implemented here
        else:
            return None
    elif project in ["mb_by_mb", "ldecc", "mb_by_mb_plus", "ldecc_plus"]:
        adj_sets: dict = eval(h["adj_sets"])
        return adj_sets.get((t1, t2), None)
    elif project in ["ldp", "ldp_plus"]:
        results: dict = eval(h["results"])
        if (t1, t2) in results:
            parts = results[(t1, t2)]
            return [
                parts["Z1"],
                parts["Z1"] + parts["Z4"],
                parts["Z1"] + parts["Z4"] + parts["Z5"],
            ]
        else:
            return None
    elif project == "local_optimal" or project.startswith("load"):
        adj_sets: dict = eval(h["adj_sets"])
        return adj_sets.get((t1, t2), None)
    else:
        raise ValueError("Invalid project")


def estimate_ates(
    results,
    algorithm: str,
    samples: dict,
    family: str = "gaussian",
) -> dict:
    ates = {}
    for h in tqdm(results, desc="Estimating ATEs"):
        exp_id = h["id"]
        ates[exp_id] = {}
        for t1, t2 in permutations(h["targets"], 2):
            adj_sets = get_adj_sets(algorithm, h, t1, t2)
            if adj_sets is None:
                ates[exp_id][(t1, t2)] = [0.0]
            else:
                ates[exp_id][(t1, t2)] = estimate_ate(
                    treatment=t1,
                    outcome=t2,
                    adj_sets=adj_sets,
                    samples=samples[exp_id],
                    family=family,
                )
    return ates


def intervention_distance(
    est_ates: dict, true_ates: dict, aggr: str = "abs"
) -> np.ndarray:
    distances = []
    for exp in tqdm(est_ates, desc="Calculating intervention distances"):
        exp_dist = []
        for pair in est_ates[exp]:
            true_ate = true_ates[exp][pair]
            if aggr == "mse":
                dist = [(true_ate - est_ate) ** 2 for est_ate in est_ates[exp][pair]]
            elif aggr == "abs":
                dist = [np.abs(true_ate - est_ate) for est_ate in est_ates[exp][pair]]
            else:
                raise ValueError("Invalid aggregation")
            exp_dist.append(np.mean(dist))
        distances.append(np.mean(exp_dist))
    return np.array(distances)


# Modified functions to evaluate on real data


def get_true_osets_real_data(experiments):
    """
    Sequentially determines the true optimal adjustment sets for a dictionary of experiments.
    """
    true_osets = {}
    for exp_id, exp in experiments.items():
        # 1. Convert the true DAG to a CPDAG
        # Using a helper (dag2cpdag) to represent the Markov Equivalence Class
        true_dag_matrix = nx.to_numpy_array(exp["true_dag"])

        # 2. Identify treatment and outcome from the true DAG
        # Based on explicit ancestral relations
        targets = exp["targets"]
        if nx.has_path(exp["true_dag"], targets[0], targets[1]):
            treatment, outcome = targets[0], targets[1]
        else:
            treatment, outcome = targets[1], targets[0]

        # 3. Retrieve the true optimal adjustment set relative to the true DAG
        true_oset = get_optimal_adj_set(true_dag_matrix, treatment, outcome)

        # 4. Store the ground truth result
        true_osets[exp_id] = {
            "treatment": treatment,
            "outcome": outcome,
            "oset": true_oset,
        }

        # print(f"Processed Experiment {exp_id}: Treatment={treatment}, Outcome={outcome}")

    return true_osets
