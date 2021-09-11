import networkx as nx
from anique import labelled_data_to_groups
from test_cmd import plot_graph
from dataset_creation import generate_sample_with_labels

if __name__ == "__main__":
    generator = generate_sample_with_labels(5, 100)
    edges, perturbed_embeddings, energy, gt_embeddings, labels = next(generator)
    G = nx.from_edgelist(edges)
    plot_graph(G, gt_embeddings, labelled_data_to_groups(labels))
    pass