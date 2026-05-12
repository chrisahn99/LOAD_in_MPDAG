
import networkx as nx
import pandas as pd
from src.running import run_experiment_mpdag_real
from sachs import fit_oracle_weights

def sanitize_to_dag(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Iteratively finds and breaks cycles in a directed graph to guarantee a strict DAG.
    It does this by removing one edge from every cycle it finds.
    """
    # Create a copy so we don't mutate the original biological graph
    safe_dag = graph.copy()
    
    cycle_count = 0
    while True:
        try:
            # Look for a cycle
            cycle = nx.find_cycle(safe_dag)
            
            # cycle is a list of edges, e.g., [(5, 51), (51, 5)]
            # We break the cycle by removing the very first edge in the list
            edge_to_remove = cycle[0]
            safe_dag.remove_edge(*edge_to_remove)
            
            cycle_count += 1
            print(f"Broke cycle by removing edge: {edge_to_remove}")
            
        except nx.NetworkXNoCycle:
            # If no cycles are found, we successfully made a DAG!
            break
            
    print(f"Sanitization complete. Removed {cycle_count} edges to ensure acyclicity.")
    return safe_dag


# The order of nodes must perfectly match the columns of your multifactorial data.
# E.g., NODE_ORDER = list(multifactorial_data.columns)  # ['G1', 'G2', ..., 'G100']

def get_dream4_ground_truth_dag(gs_dataframe: pd.DataFrame, node_order: list) -> nx.DiGraph:
    """
    Constructs the ground truth DAG for the DREAM4 dataset.
    Nodes are integer indices corresponding to the order in node_order.
    """
    # 1. Create the reverse mapping (string name -> integer index)
    name_to_idx = {name: idx for idx, name in enumerate(node_order)}

    true_dag = nx.DiGraph()

    # 2. Add ALL nodes as integer indices (0 through len(node_order)-1)
    # This ensures disconnected genes still exist mathematically in your graph
    true_dag.add_nodes_from(range(len(node_order)))

    # 3. Extract true edges and map string names to integer indices
    edges_by_index = []
    
    # We assume your dataframe was loaded with columns: ['cause', 'outcome', 'edge']
    for _, row in gs_dataframe.iterrows():
        cause_name = row['cause']
        outcome_name = row['outcome']
        edge_exists = row['edge'] 
        
        # In DREAM4, the file lists non-existent edges as 0. We ONLY want the 1s.
        if edge_exists == 1:
            # Safety check to ensure the nodes exist in our multifactorial dataset
            if cause_name in name_to_idx and outcome_name in name_to_idx:
                u = name_to_idx[cause_name]
                v = name_to_idx[outcome_name]
                edges_by_index.append((u, v))

    # 4. Add the integer edges to the graph
    true_dag.add_edges_from(edges_by_index)

    return true_dag




def run_experiment_dream4(subset_number, sampling_strategy="local", frac_req=0.3):

    # 1. Load your multifactorial data
    multifactorial_df = pd.read_csv(f"experiments/dream4/insilico_size100_{subset_number}_multifactorial.tsv", sep='\t')
    NODE_ORDER = list(multifactorial_df.columns)

    # 2. Load the DREAM4 Gold Standard and give it headers
    dream4_gs = pd.read_csv(f"experiments/dream4/DREAM4_GoldStandard_InSilico_Size100_multifactorial_{subset_number}.tsv", 
                            sep='\t', header=None, names=['cause', 'outcome', 'edge'])

    # 3. Generate the true binary DAG
    binary_dag = sanitize_to_dag(get_dream4_ground_truth_dag(dream4_gs, NODE_ORDER))

    # 4. Fit the Oracle Weights to get your linear SEM
    dream4_data_array = multifactorial_df.to_numpy()
    oracle_dag = fit_oracle_weights(binary_dag, dream4_data_array)

    forward_pairs_idx = []
    for node in binary_dag.nodes():
        # nx.descendants finds all nodes downstream (direct and indirect)
        for descendant in nx.descendants(binary_dag, node):
            forward_pairs_idx.append([node, descendant])


    base_args_skeleton = {
        "data_name": f"dream4_net{subset_number}",
        "real_data": dream4_data_array,
        "true_dag": oracle_dag,
        "observed": dream4_data_array.shape[1],
        "alpha": 0.01,
        "ci_test": "fisherz",
        "sampling_strategy": sampling_strategy
    }

    base_seed = 42
    frac_forb = 0

    base_args_skeleton["fraction_forbidden"] = frac_forb
    base_args_skeleton["fraction_required"] = frac_req

    # Run experiment
    result_dict_dream4, details_dict_dream4 = run_experiment_mpdag_real(base_args_skeleton, base_seed, forward_pairs_idx)
    return result_dict_dream4, details_dict_dream4



# You MUST define this list so it exactly matches the column order of your 
# filtered 99-column expression dataset! 
# (e.g., TF_ORDER = ['G1', 'G2', 'G3', ..., 'G99'])
# You can usually get this from: tf_order = list(filtered_expression_df.columns)

def get_dream5_tf_ground_truth_dag(gs_dataframe: pd.DataFrame, tf_order: list) -> nx.DiGraph:
    """
    Constructs the ground truth DAG for the DREAM5 dataset, restricted to a specific list of TFs.
    Nodes are integer indices corresponding to the order in tf_order.
    """
    # 1. Create the reverse mapping (name -> index)
    name_to_idx = {name: idx for idx, name in enumerate(tf_order)}

    true_dag = nx.DiGraph()

    # 2. Add ALL nodes as integer indices (0 through 98)
    # CRITICAL: This ensures disconnected TFs still exist in the math!
    true_dag.add_nodes_from(range(len(tf_order)))

    # 3. Extract edges and map string names to integer indices
    edges_by_index = []
    for _, row in gs_dataframe.iterrows():
        cause_name = row['cause']
        outcome_name = row['outcome']
        
        # Safety check: Only add the edge if BOTH nodes are in our 99 TF list
        if cause_name in name_to_idx and outcome_name in name_to_idx:
            u = name_to_idx[cause_name]
            v = name_to_idx[outcome_name]
            edges_by_index.append((u, v))

    # 4. Add the integer edges to the graph
    true_dag.add_edges_from(edges_by_index)

    return true_dag


def test_dream5(sampling_strategy, frac_req):
    base_args_skeleton = {
        "data_name": "dream5_net2",
        "real_data": filtered_expression_data.to_numpy(),
        "true_dag": oracle_dag_dream5,
        "observed": filtered_expression_data.to_numpy().shape[1],
        "alpha": 0.01,
        "ci_test": "fisherz",
        "sampling_strategy": sampling_strategy
    }

    base_seed = 42
    frac_forb = 0
    frac_req = frac_req

    base_args_skeleton["fraction_forbidden"] = frac_forb
    base_args_skeleton["fraction_required"] = frac_req

    # Run experiment
    result_dict_dream5, details_dict_dream5 = run_experiment_mpdag_real(base_args_skeleton, base_seed, forward_pairs_idx)
    return result_dict_dream5, details_dict_dream5

for sampling_strategy in tqdm(["local", "global"]):
    for frac_req in tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        print(f"Testing sampling_strategy={sampling_strategy}, frac_req={frac_req}")
        result_dict, details_dict = test_dream5(sampling_strategy, frac_req)
        print(f"Results: {result_dict}\n")