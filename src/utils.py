import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_ground_truth(experiments, exp_id):
  true_dag = experiments[exp_id]["true_dag"]
  nx.draw_circular(true_dag, with_labels=True)


def draw_ground_truth_from_dag(true_dag):
  nx.draw_circular(true_dag, with_labels=True)



def draw_graph_from_array(array):
    plt.figure() # Creates a fresh window/figure for this specific graph
    nx.draw_circular(nx.DiGraph(array), with_labels=True)
    plt.show()   # Ensures the plot is rendered before moving to the next block


def draw_complex_graph(array):
    arr = np.array(array)
    G = nx.DiGraph()
    G.add_nodes_from(range(len(arr)))

    # Identify edges based on your logic
    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i, j] == 1:
                G.add_edge(j, i) # Arrow from j to i

    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 6))

    # Draw nodes and directed edges (the 1s)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrows=True)

    plt.show()


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