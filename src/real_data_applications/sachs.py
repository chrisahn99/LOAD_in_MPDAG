import networkx as nx
from sklearn.linear_model import LinearRegression

# The strict ordering of proteins corresponding to your dataset columns
SACHS_PROTEIN_ORDER = [
    'Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3',
    'Erk', 'Akt', 'PKA', 'PKC', 'P38', 'Jnk'
]


# [Treatment, Outcome]
forward_pairs = [
    # From Plcg
    ['Plcg', 'PIP3'], ['Plcg', 'PIP2'], ['Plcg', 'PKC'], ['Plcg', 'PKA'], ['Plcg', 'Raf'], ['Plcg', 'Mek'], ['Plcg', 'Erk'], ['Plcg', 'Akt'], ['Plcg', 'P38'], ['Plcg', 'Jnk'],
    # From PIP3
    ['PIP3', 'PIP2'], ['PIP3', 'PKC'], ['PIP3', 'PKA'], ['PIP3', 'Raf'], ['PIP3', 'Mek'], ['PIP3', 'Erk'], ['PIP3', 'Akt'], ['PIP3', 'P38'], ['PIP3', 'Jnk'],
    # From PIP2
    ['PIP2', 'PKC'], ['PIP2', 'PKA'], ['PIP2', 'Raf'], ['PIP2', 'Mek'], ['PIP2', 'Erk'], ['PIP2', 'Akt'], ['PIP2', 'P38'], ['PIP2', 'Jnk'],
    # From PKC
    ['PKC', 'PKA'], ['PKC', 'Raf'], ['PKC', 'Mek'], ['PKC', 'Erk'], ['PKC', 'Akt'], ['PKC', 'P38'], ['PKC', 'Jnk'],
    # From PKA
    ['PKA', 'Raf'], ['PKA', 'Mek'], ['PKA', 'Erk'], ['PKA', 'Akt'], ['PKA', 'P38'], ['PKA', 'Jnk'],
    # From Raf
    ['Raf', 'Mek'], ['Raf', 'Erk'], ['Raf', 'Akt'],
    # From Mek
    ['Mek', 'Erk'], ['Mek', 'Akt'],
    # From Erk
    ['Erk', 'Akt']
]

# [Outcome, Treatment] (Inverse direction)
inverse_pairs = [
    # To Plcg
    ['PIP3', 'Plcg'], ['PIP2', 'Plcg'], ['PKC', 'Plcg'], ['PKA', 'Plcg'], ['Raf', 'Plcg'], ['Mek', 'Plcg'], ['Erk', 'Plcg'], ['Akt', 'Plcg'], ['P38', 'Plcg'], ['Jnk', 'Plcg'],
    # To PIP3
    ['PIP2', 'PIP3'], ['PKC', 'PIP3'], ['PKA', 'PIP3'], ['Raf', 'PIP3'], ['Mek', 'PIP3'], ['Erk', 'PIP3'], ['Akt', 'PIP3'], ['P38', 'PIP3'], ['Jnk', 'PIP3'],
    # To PIP2
    ['PKC', 'PIP2'], ['PKA', 'PIP2'], ['Raf', 'PIP2'], ['Mek', 'PIP2'], ['Erk', 'PIP2'], ['Akt', 'PIP2'], ['P38', 'PIP2'], ['Jnk', 'PIP2'],
    # To PKC
    ['PKA', 'PKC'], ['Raf', 'PKC'], ['Mek', 'PKC'], ['Erk', 'PKC'], ['Akt', 'PKC'], ['P38', 'PKC'], ['Jnk', 'PKC'],
    # To PKA
    ['Raf', 'PKA'], ['Mek', 'PKA'], ['Erk', 'PKA'], ['Akt', 'PKA'], ['P38', 'PKA'], ['Jnk', 'PKA'],
    # To Raf
    ['Mek', 'Raf'], ['Erk', 'Raf'], ['Akt', 'Raf'],
    # To Mek
    ['Erk', 'Mek'], ['Akt', 'Mek'],
    # To Erk
    ['Akt', 'Erk']
]


# Helper to quickly generate the integer lists if needed
name_to_idx = {name: idx for idx, name in enumerate(SACHS_PROTEIN_ORDER)}

forward_pairs_idx = [[name_to_idx[t], name_to_idx[o]] for t, o in forward_pairs]
inverse_pairs_idx = [[name_to_idx[o], name_to_idx[t]] for o, t in inverse_pairs]

# The resulting forward_pairs_idx evaluates to:
# [
#     [2, 4], [2, 3], [2, 8], [2, 7], [2, 0], [2, 1], [2, 5], [2, 6], [2, 9], [2, 10],
#     [4, 3], [4, 8], [4, 7], [4, 0], [4, 1], [4, 5], [4, 6], [4, 9], [4, 10],
#     [3, 8], [3, 7], [3, 0], [3, 1], [3, 5], [3, 6], [3, 9], [3, 10],
#     [8, 7], [8, 0], [8, 1], [8, 5], [8, 6], [8, 9], [8, 10],
#     [7, 0], [7, 1], [7, 5], [7, 6], [7, 9], [7, 10],
#     [0, 1], [0, 5], [0, 6],
#     [1, 5], [1, 6],
#     [5, 6]
# ]


def get_protein_name(index: int) -> str:
    """
    Maps an integer index back to the actual protein name.
    """
    if 0 <= index < len(SACHS_PROTEIN_ORDER):
        return SACHS_PROTEIN_ORDER[index]
    raise IndexError(f"Index {index} is out of bounds for the Sachs protein list.")

def get_sachs_ground_truth_dag() -> nx.DiGraph:
    """
    Constructs the ground truth DAG for the Sachs protein dataset.
    Nodes are integer indices corresponding to SACHS_PROTEIN_ORDER.
    """
    # Create a reverse mapping (name -> index) to make building the edges less error-prone
    name_to_idx = {name: idx for idx, name in enumerate(SACHS_PROTEIN_ORDER)}

    true_dag = nx.DiGraph()

    # 1. Define nodes as integer indices (0 through 10)
    true_dag.add_nodes_from(range(len(SACHS_PROTEIN_ORDER)))

    # 2. Define edges using string names for clarity, mapped directly from the ground truth
    edges_by_name = [
        # From Plcg
        ('Plcg', 'PIP3'),
        ('Plcg', 'PIP2'),
        ('Plcg', 'PKC'),

        # From PIP3
        ('PIP3', 'PIP2'),
        ('PIP3', 'Akt'),

        # From PIP2
        ('PIP2', 'PKC'),

        # From PKC
        ('PKC', 'PKA'),
        ('PKC', 'Raf'),
        ('PKC', 'Mek'),
        ('PKC', 'Jnk'),
        ('PKC', 'P38'),

        # From PKA
        ('PKA', 'Raf'),
        ('PKA', 'Mek'),
        ('PKA', 'Erk'),
        ('PKA', 'Akt'),
        ('PKA', 'Jnk'),
        ('PKA', 'P38'),

        # Downstream cascade
        ('Raf', 'Mek'),
        ('Mek', 'Erk'),
        ('Erk', 'Akt')
    ]

    # 3. Convert named edges to integer index edges and add to graph
    edges_by_index = [(name_to_idx[u], name_to_idx[v]) for u, v in edges_by_name]
    true_dag.add_edges_from(edges_by_index)

    return true_dag


def fit_oracle_weights(true_dag: nx.DiGraph, real_samples: np.ndarray) -> nx.DiGraph:
    """
    Fits linear SEM weights to the true DAG using real observational data.
    This creates the 'Oracle' ground truth for your intervention distance.
    
    Args:
        true_dag: The binary ground truth DAG (nodes must correspond to column indices).
        real_samples: The actual data loaded from causal-learn.
        
    Returns:
        A weighted nx.DiGraph where edges contain the fitted linear coefficients.
    """
    # Create a copy so we don't mutate the original binary topology
    oracle_dag = true_dag.copy()

    # Iterate through every node to calculate incoming edge weights
    for node in oracle_dag.nodes():
        parents = list(oracle_dag.predecessors(node))
        
        # If the node has no incoming arrows, there are no weights to assign
        if not parents:
            continue
            
        # Extract predictors (true parents) and target (the current node)
        X = real_samples[:, parents]
        y = real_samples[:, node]

        # Fit an Ordinary Least Squares regression
        # We include an intercept to match the behavior of your estimate_ate function
        reg = LinearRegression(fit_intercept=True).fit(X, y)

        # Assign the fitted coefficients as the 'weight' attribute for each incoming edge
        for i, parent in enumerate(parents):
            oracle_dag[parent][node]['weight'] = reg.coef_[i]

    return oracle_dag