from time import perf_counter

import numpy as np
import networkx as nx
from tqdm import tqdm

from src.counting_ci_tests import CountingTest
from src.load import load, load_in_mpdag
from src.data_generation import generate_data
from src.background_knowledge import (
    sample_local_background_knowledge,
    sample_background_knowledge_v2,
    initialize_background_knowledge,
    sample_local_background_knowledge_noised
)
from src.evaluation import get_true_osets, evaluate_oset, true_causal_effects, evaluate_intervention

### Sampling around target nodes only

def run_algorithm_mpdag(
    id: str,
    data: np.ndarray,
    targets: list[int],
    ci_test: str,
    alpha: float,
    with_b: bool = False,
    bg_knowledge: dict | None = None,
    true_dag: nx.DiGraph | None = None,
    cpt: dict | None = None,
    logging: bool = False,
    **kwargs,
  ):
  ci_test = CountingTest(data, ci_test, **kwargs)


  start = perf_counter()
  # Run algorithm
  if with_b:
      result = load_in_mpdag(
          data=data,
          ci_test=ci_test,
          alpha=alpha,
          targets=targets,
          background_knowledge=bg_knowledge,
          mb_algorithm="grow_shrink",
          logging=logging,
          **kwargs
      )
  else:
      result = load(
          data=data,
          ci_test=ci_test,
          alpha=alpha,
          targets=targets,
          mb_algorithm="grow_shrink",
          logging=logging,
          **kwargs
      )

  result["time"] = perf_counter() - start
  result["id"] = id
  result["targets"] = targets
  result["tests"] = ci_test.get_tests_per_order().tolist()
  result["cpt"] = cpt
  result["true_dag"] = true_dag
  result["data"] = data

  return result


## Modification March 17th: using bg sampler v2 (around 4pm)
## New modification March 17th: using local sampler again


def get_experiment_mpdag(
      seed: int,
      observed: int,
      exp_degree: float,
      max_degree: int,
      targets: int,
      expl_anc: bool,
      identifiable: bool,
      min_adj_size: int,
      samples_num,
      discrete: bool,
      save: bool,
      with_b: bool,
      logging: bool,
      fraction_forbidden: float = 0.1,
      fraction_required: float = 0.1,
      alpha: float = 0.01,
      ci_test: str = "fisherz",
    ) -> dict:
    """
    Run a single experiment with a specific seed.

    Args:
        args (Namespace): The command line arguments.
        done (dict): A dictionary of completed experiments.
        s (int): The seed for the experiment.

    Returns:
        dict: The result of the experiment.
    """

    # Generate the synthetic data
    data = generate_data(
        seed=seed,
        observed=observed,        # Slightly more nodes to ensure a richer MB
        exp_degree=exp_degree,
        max_degree=max_degree,
        targets=targets,
        identifiable=identifiable,
        min_adj_size=min_adj_size,
        samples_num=samples_num,   # More samples for more reliable Fisher-Z tests
        expl_anc=expl_anc,
        discrete=discrete,
        save=save
    )


    # Extract components
    true_dag = data["true_dag"]
    cpt = data["cpt"]
    targets = data["targets"]


    # Sample background knowledge (e.g., 20% of edges)
    bg_knowledge = sample_local_background_knowledge(true_dag, targets, fraction_required, fraction_forbidden, seed)

    initial_G = initialize_background_knowledge(num_nodes=observed, bk_dict=bg_knowledge)

    return run_algorithm_mpdag(
        ci_test=ci_test,
        alpha=alpha,
        with_b=with_b,
        bg_knowledge=initial_G,
        logging=logging,
        **data
    )


def run_experiment_mpdag(base_args, base_seed, n_exp):

  experiments_with_b = {}
  experiments_without_b = {}

  for s in tqdm(range(n_exp)):
    exp_with_b = get_experiment_mpdag(**base_args, seed=base_seed+s, with_b=True)
    experiments_with_b[exp_with_b["id"]] = exp_with_b

    exp_without_b = get_experiment_mpdag(**base_args, seed=base_seed+s, with_b=False)
    experiments_without_b[exp_without_b["id"]] = exp_without_b

  # Global metrics
  true_osets_with_b = get_true_osets(experiments_with_b)
  f1_with_b = np.sort(evaluate_oset("load", list(experiments_with_b.values()), true_osets_with_b)[2])

  true_osets_without_b = get_true_osets(experiments_without_b)
  f1_without_b = np.sort(evaluate_oset("load", list(experiments_without_b.values()), true_osets_without_b)[2])

  # Intervention metrics
  family = "binary" if base_args.get("discrete", False) else "gaussian"
  true_effects = true_causal_effects(experiments_with_b, family=family)

  int_dist_with_b = evaluate_intervention("load", list(experiments_with_b.values()), true_effects, family=family)
  int_dist_without_b = evaluate_intervention("load", list(experiments_without_b.values()), true_effects, family=family)

  f1_dict = {
      "f1_with_b": {"mean":f1_with_b.mean(), "std": f1_with_b.std()},
      "f1_without_b": {"mean":f1_without_b.mean(), "std": f1_without_b.std()},
      "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
      "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
  }

  return f1_dict


def run_experiment_mpdag_light(base_args, base_seed, n_exp):
    # Lighter version that doesn't store all experiments in memory (only the final metrics), to save memory when running many experiments

    f1_with_b_list = []
    f1_without_b_list = []
    int_dist_with_b_list = []
    int_dist_without_b_list = []
    family = "binary" if base_args.get("discrete", False) else "gaussian"

    for s in tqdm(range(n_exp)):
        exp_with_b = get_experiment_mpdag(**base_args, seed=base_seed+s, with_b=True)
        exp_without_b = get_experiment_mpdag(**base_args, seed=base_seed+s, with_b=False)

        # Evaluate immediately, store only scalars
        true_osets_with_b = get_true_osets({exp_with_b["id"]: exp_with_b})
        f1_with_b_list.append(evaluate_oset("load", [exp_with_b], true_osets_with_b)[2])

        true_osets_without_b = get_true_osets({exp_without_b["id"]: exp_without_b})
        f1_without_b_list.append(evaluate_oset("load", [exp_without_b], true_osets_without_b)[2])

        true_effects = true_causal_effects({exp_with_b["id"]: exp_with_b}, family=family)
        int_dist_with_b_list.append(evaluate_intervention("load", [exp_with_b], true_effects, family=family))
        int_dist_without_b_list.append(evaluate_intervention("load", [exp_without_b], true_effects, family=family))

        # Explicitly discard large objects
        del exp_with_b, exp_without_b

    f1_with_b = np.sort(np.concatenate(f1_with_b_list))
    f1_without_b = np.sort(np.concatenate(f1_without_b_list))
    int_dist_with_b = np.concatenate(int_dist_with_b_list)
    int_dist_without_b = np.concatenate(int_dist_without_b_list)

    return {
        "f1_with_b": {"mean": f1_with_b.mean(), "std": f1_with_b.std()},
        "f1_without_b": {"mean": f1_without_b.mean(), "std": f1_without_b.std()},
        "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
        "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
    }


### V2: sampling around all nodes


def get_experiment_mpdag_v2(
      seed: int,
      observed: int,
      exp_degree: float,
      max_degree: int,
      targets: int,
      expl_anc: bool,
      identifiable: bool,
      min_adj_size: int,
      samples_num,
      discrete: bool,
      save: bool,
      with_b: bool,
      logging: bool,
      fraction_forbidden: float = 0.1,
      fraction_required: float = 0.1,
      alpha: float = 0.01,
      ci_test: str = "fisherz",
    ) -> dict:
    """
    Run a single experiment with a specific seed.

    Args:
        args (Namespace): The command line arguments.
        done (dict): A dictionary of completed experiments.
        s (int): The seed for the experiment.

    Returns:
        dict: The result of the experiment.
    """

    # Generate the synthetic data
    data = generate_data(
        seed=seed,
        observed=observed,        # Slightly more nodes to ensure a richer MB
        exp_degree=exp_degree,
        max_degree=max_degree,
        targets=targets,
        identifiable=identifiable,
        min_adj_size=min_adj_size,
        samples_num=samples_num,   # More samples for more reliable Fisher-Z tests
        expl_anc=expl_anc,
        discrete=discrete,
        save=save
    )


    # Extract components
    true_dag = data["true_dag"]
    cpt = data["cpt"]
    targets = data["targets"]


    # Sample background knowledge (e.g., 20% of edges)
    bg_knowledge = sample_background_knowledge_v2(true_dag, fraction_required, fraction_forbidden, seed)

    initial_G = initialize_background_knowledge(num_nodes=observed, bk_dict=bg_knowledge)

    return run_algorithm_mpdag(
        ci_test=ci_test,
        alpha=alpha,
        with_b=with_b,
        bg_knowledge=initial_G,
        logging=logging,
        **data
    )


def run_experiment_mpdag_v2(base_args, base_seed, n_exp):

  experiments_with_b = {}
  experiments_without_b = {}

  for s in tqdm(range(n_exp)):
    exp_with_b = get_experiment_mpdag_v2(**base_args, seed=base_seed+s, with_b=True)
    experiments_with_b[exp_with_b["id"]] = exp_with_b

    exp_without_b = get_experiment_mpdag_v2(**base_args, seed=base_seed+s, with_b=False)
    experiments_without_b[exp_without_b["id"]] = exp_without_b

  # Global metrics
  true_osets_with_b = get_true_osets(experiments_with_b)
  f1_with_b = np.sort(evaluate_oset("load", list(experiments_with_b.values()), true_osets_with_b)[2])

  true_osets_without_b = get_true_osets(experiments_without_b)
  f1_without_b = np.sort(evaluate_oset("load", list(experiments_without_b.values()), true_osets_without_b)[2])

  # Intervention metrics
  family = "binary" if base_args.get("discrete", False) else "gaussian"
  true_effects = true_causal_effects(experiments_with_b, family=family)

  int_dist_with_b = evaluate_intervention("load", list(experiments_with_b.values()), true_effects, family=family)
  int_dist_without_b = evaluate_intervention("load", list(experiments_without_b.values()), true_effects, family=family)

  f1_dict = {
      "f1_with_b": {"mean":f1_with_b.mean(), "std": f1_with_b.std()},
      "f1_without_b": {"mean":f1_without_b.mean(), "std": f1_without_b.std()},
      "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
      "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
  }

  return f1_dict


def run_experiment_mpdag_v2_light(base_args, base_seed, n_exp):
    # Lighter version that doesn't store all experiments in memory (only the final metrics), to save memory when running many experiments

    f1_with_b_list = []
    f1_without_b_list = []
    int_dist_with_b_list = []
    int_dist_without_b_list = []
    family = "binary" if base_args.get("discrete", False) else "gaussian"

    for s in tqdm(range(n_exp)):
        exp_with_b = get_experiment_mpdag_v2(**base_args, seed=base_seed+s, with_b=True)
        exp_without_b = get_experiment_mpdag_v2(**base_args, seed=base_seed+s, with_b=False)

        # Evaluate immediately, store only scalars
        true_osets_with_b = get_true_osets({exp_with_b["id"]: exp_with_b})
        f1_with_b_list.append(evaluate_oset("load", [exp_with_b], true_osets_with_b)[2])

        true_osets_without_b = get_true_osets({exp_without_b["id"]: exp_without_b})
        f1_without_b_list.append(evaluate_oset("load", [exp_without_b], true_osets_without_b)[2])

        true_effects = true_causal_effects({exp_with_b["id"]: exp_with_b}, family=family)
        int_dist_with_b_list.append(evaluate_intervention("load", [exp_with_b], true_effects, family=family))
        int_dist_without_b_list.append(evaluate_intervention("load", [exp_without_b], true_effects, family=family))

        # Explicitly discard large objects
        del exp_with_b, exp_without_b

    f1_with_b = np.sort(np.concatenate(f1_with_b_list))
    f1_without_b = np.sort(np.concatenate(f1_without_b_list))
    int_dist_with_b = np.concatenate(int_dist_with_b_list)
    int_dist_without_b = np.concatenate(int_dist_without_b_list)

    return {
        "f1_with_b": {"mean": f1_with_b.mean(), "std": f1_with_b.std()},
        "f1_without_b": {"mean": f1_without_b.mean(), "std": f1_without_b.std()},
        "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
        "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
    }


### Real data (sachs)


def get_experiment_mpdag_real(
      data_name: str,
      seed: int,
      real_data: np.ndarray,
      true_dag: nx.DiGraph,
      targets: list[int],
      observed: int,
      with_b: bool,
      fraction_forbidden: float = 0.1,
      fraction_required: float = 0.1,
      alpha: float = 0.01,
      ci_test: str = "fisherz",
    ) -> dict:
    """
    Run a single experiment with a specific seed.

    Args:
        args (Namespace): The command line arguments.
        done (dict): A dictionary of completed experiments.
        s (int): The seed for the experiment.

    Returns:
        dict: The result of the experiment.
    """



    # Sample background knowledge (e.g., 20% of edges)
    bg_knowledge = sample_background_knowledge_v2(true_dag, fraction_required, fraction_forbidden, seed)

    initial_G = initialize_background_knowledge(num_nodes=observed, bk_dict=bg_knowledge)
    id = f"{data_name}_{seed}"

    return run_algorithm_mpdag(
        id=id,
        targets=targets,
        data=real_data,
        ci_test=ci_test,
        alpha=alpha,
        with_b=with_b,
        bg_knowledge=initial_G,
        true_dag=true_dag,
    )


def run_experiment_mpdag_real(base_args, base_seed, pairs_to_test):

  experiments_with_b = {}
  experiments_without_b = {}

  for s, pair in enumerate(tqdm(pairs_to_test)):
    exp_with_b = get_experiment_mpdag_real(**base_args, targets=pair, seed=base_seed+s, with_b=True)
    experiments_with_b[exp_with_b["id"]] = exp_with_b

    exp_without_b = get_experiment_mpdag_real(**base_args, targets=pair, seed=base_seed+s, with_b=False)
    experiments_without_b[exp_with_b["id"]] = exp_without_b

  # Global metrics
  true_osets_with_b = get_true_osets_real_data(experiments_with_b)
  res_with = evaluate_oset("load", list(experiments_with_b.values()), true_osets_with_b)
  # Convert to numpy arrays immediately
  prec_with_b, rec_with_b, f1_with_b = [np.array(x) for x in res_with]

  true_osets_without_b = get_true_osets_real_data(experiments_without_b)
  res_without = evaluate_oset("load", list(experiments_without_b.values()), true_osets_without_b)
  # Convert to numpy arrays immediately
  prec_without_b, rec_without_b, f1_without_b = [np.array(x) for x in res_without]


  result_dict = {
      "precision_with_b": {"mean": np.array(prec_with_b).mean(), "std": prec_with_b.std()},
      "recall_with_b": {"mean": rec_with_b.mean(), "std": rec_with_b.std()},
      "f1_with_b": {"mean": f1_with_b.mean(), "std": f1_with_b.std()},

      "precision_without_b": {"mean": prec_without_b.mean(), "std": prec_without_b.std()},
      "recall_without_b": {"mean": rec_without_b.mean(), "std": rec_without_b.std()},
      "f1_without_b": {"mean": f1_without_b.mean(), "std": f1_without_b.std()},
  }

  # Detailed dictionary (Raw data per iteration)
  detailed_dict = {
      "with_b": {
          "precision": prec_with_b,
          "recall": rec_with_b,
          "f1": f1_with_b
      },
      "without_b": {
          "precision": prec_without_b,
          "recall": rec_without_b,
          "f1": f1_without_b
      }
  }

  return result_dict, detailed_dict


### Noised: testing robustness to noise in the background knowledge (sampling around target nodes, but with noise)


def get_experiment_mpdag_noised(
      seed: int,
      observed: int,
      exp_degree: float,
      max_degree: int,
      targets: int,
      expl_anc: bool,
      identifiable: bool,
      min_adj_size: int,
      samples_num,
      discrete: bool,
      save: bool,
      with_b: bool,
      logging: bool,
      fraction_forbidden: float = 0.1,
      fraction_required: float = 0.1,
      false_required_ratio: float = 0.5,
      alpha: float = 0.01,
      ci_test: str = "fisherz",
    ) -> dict:
    """
    Run a single experiment with a specific seed.

    Args:
        args (Namespace): The command line arguments.
        done (dict): A dictionary of completed experiments.
        s (int): The seed for the experiment.

    Returns:
        dict: The result of the experiment.
    """

    # Generate the synthetic data
    data = generate_data(
        seed=seed,
        observed=observed,        # Slightly more nodes to ensure a richer MB
        exp_degree=exp_degree,
        max_degree=max_degree,
        targets=targets,
        identifiable=identifiable,
        min_adj_size=min_adj_size,
        samples_num=samples_num,   # More samples for more reliable Fisher-Z tests
        expl_anc=expl_anc,
        discrete=discrete,
        save=save
    )


    # Extract components
    true_dag = data["true_dag"]
    cpt = data["cpt"]
    targets = data["targets"]


    # Sample background knowledge (e.g., 20% of edges)
    bg_knowledge = sample_local_background_knowledge_noised(
        true_dag=true_dag,
        targets=targets,
        fraction_required=fraction_required,
        fraction_forbidden=fraction_forbidden,
        noise_mu=false_required_ratio,
        seed=seed
    )
    
    initial_G = initialize_background_knowledge(num_nodes=observed, bk_dict=bg_knowledge)

    return run_algorithm_mpdag(
        ci_test=ci_test,
        alpha=alpha,
        with_b=with_b,
        bg_knowledge=initial_G,
        logging=logging,
        **data
    )


def run_experiment_mpdag_noised(base_args, base_seed, n_exp):

  experiments_with_b = {}
  experiments_without_b = {}

  for s in tqdm(range(n_exp)):
    exp_with_b = get_experiment_mpdag_noised(**base_args, seed=base_seed+s, with_b=True)
    experiments_with_b[exp_with_b["id"]] = exp_with_b

    exp_without_b = get_experiment_mpdag_noised(**base_args, seed=base_seed+s, with_b=False)
    experiments_without_b[exp_without_b["id"]] = exp_without_b

  # Global metrics
  true_osets_with_b = get_true_osets(experiments_with_b)
  f1_with_b = np.sort(evaluate_oset("load", list(experiments_with_b.values()), true_osets_with_b)[2])

  true_osets_without_b = get_true_osets(experiments_without_b)
  f1_without_b = np.sort(evaluate_oset("load", list(experiments_without_b.values()), true_osets_without_b)[2])

  # Intervention metrics
  family = "binary" if base_args.get("discrete", False) else "gaussian"
  true_effects = true_causal_effects(experiments_with_b, family=family)

  int_dist_with_b = evaluate_intervention("load", list(experiments_with_b.values()), true_effects, family=family)
  int_dist_without_b = evaluate_intervention("load", list(experiments_without_b.values()), true_effects, family=family)

  f1_dict = {
      "f1_with_b": {"mean":f1_with_b.mean(), "std": f1_with_b.std()},
      "f1_without_b": {"mean":f1_without_b.mean(), "std": f1_without_b.std()},
      "int_dist_with_b": {"mean": int_dist_with_b.mean(), "std": int_dist_with_b.std()},
      "int_dist_without_b": {"mean": int_dist_without_b.mean(), "std": int_dist_without_b.std()}
  }

  return f1_dict