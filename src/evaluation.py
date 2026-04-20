import os

import numpy as np
import networkx as nx
from rpy2.robjects import default_converter, globalenv, numpy2ri, r
from rpy2 import robjects

# 1. Define the absolute path to your shared folder shortcut or MyDrive folder
# Use the 'Copy Path' feature in the Colab sidebar to get this exactly right

# replace with file path on ruche server
file_path = "/workdir/ahns/evaluate.R"

# 2. Source the file using the R 'source' function via robjects
if os.path.exists(file_path):
    robjects.r.source(file_path)
    print(f"Successfully sourced: {file_path}")
else:
    print(f"Error: File not found at {file_path}")


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



# INTERVENTION DISTANCE
def is_ancestor(t1: int, t2: int, amat: np.ndarray) -> bool:
    reach = amat.copy().astype(bool)
    np.fill_diagonal(reach, True)
    reach = np.linalg.matrix_power(reach, reach.shape[0] - 1)
    return reach[t1, t2]


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

    # 4. Return trimmed results (best 5 and worst 5 removed)
    # This follows the paper's methodology to show general trends
    return (
        (prec_scores),
        (rec_scores),
        (f1_scores),
    )


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