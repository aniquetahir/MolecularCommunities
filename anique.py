import pickle
from collections import defaultdict
import numpy as np
import networkx as nx
from typing import Dict, Union
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE, Isomap
import networkx as nx
import random
import matplotlib.pyplot as plt
from functools import partial
from itertools import combinations
from scipy.optimize import fsolve, least_squares
from graspologic.simulations import sbm


def plot_graph(G, embeddings, communities):
    pos_md = dict(zip(range(len(embeddings)), np.array(embeddings)))
    cmap = plt.get_cmap(lut=len(communities))
    colors = list(cmap.colors)
    random.shuffle(colors)
    for i, community in enumerate(communities):
        c = [colors[i]] * len(community)
        nx.draw_networkx_nodes(G, pos_md, nodelist=community, node_color=c, node_size=7)
    nx.draw_networkx_edges(G, pos_md, alpha=0.5)
    # nx.draw_networkx_nodes(G, pos_md, nodelist=communities[0], node_color='r')
    # nx.draw_networkx_nodes(G, pos_md, nodelist=communities[1], node_color='g')
    # if len(communities) > 2:
    #     nx.draw_networkx_nodes(G, pos_md, nodelist=communities[2], node_color='b')
    # nx.draw_networkx_edges(G, pos_md)
    plt.show()
    # input("Press enter to continue")


def get_reduced_sample_graph(G, embeddings, labels, sample_size):
    original_nodes = list(G.nodes)
    num_nodes = G.number_of_nodes()
    if num_nodes < sample_size:
        raise Warning(f'Sample size({sample_size}) greater than number of nodes({num_nodes})')
    deletion_nodes = random.sample(G.nodes, max(0, num_nodes - sample_size))
    G.remove_nodes_from(deletion_nodes)
    new_nodes = list(G.nodes) # Careful. After readjust, new_nodes is useless
    new_G, index_map = readjust_graph(G)
    # remove useless nodes from labels
    new_labels = []
    new_embeddings = []
    for i, n in enumerate(original_nodes):
        if n in new_nodes:
            new_labels.append(labels[i])
            new_embeddings.append(embeddings[i])
    # labels = new_labels
    num_nodes = new_G.number_of_nodes()
    return new_G, embeddings, labels, index_map


def plot_reduced_embedding(G: nx.Graph, embeddings, labels, method: Union[PCA.__class__]=PCA):
    groups = labelled_data_to_groups(labels)
    dr_model = method(2)
    reduced_embedding = dr_model.fit_transform(embeddings)
    plot_graph(G, reduced_embedding, groups)


def load_pickle(filepath: str):
    with open(filepath, 'rb') as pkl_file:
        obj = pickle.load(pkl_file)
    return obj


def save_pickle(obj, filepath: str):
    with open(filepath, 'wb') as pkl_file:
        pickle.dump(obj, pkl_file)


def onehot_to_cat(mat):
    labels = np.argmax(mat, axis=1)
    return np.array(labels).flatten()


def labelled_data_to_groups(labels):
    group_dict = defaultdict(list)
    for i, label in enumerate(labels):
        group_dict[label].append(i)
    return list(group_dict.values())


def opposite_dict(d: Dict):
    o = {}
    for k, v in d.items():
        o[v] = k
    return o


def readjust_graph(G: nx.Graph, suggestions={}):
    nodes = list(G.nodes)
    index_map = suggestions
    reverse_map = opposite_dict(suggestions)
    for i, x in enumerate(nodes):
        if x not in reverse_map.keys():
            index_map[i] = x
            reverse_map[x] = i

    new_edge_list = []
    for a, b in G.edges:
        new_edge_list.append((reverse_map[a], reverse_map[b]))
    new_G = nx.Graph()
    new_G.add_nodes_from([reverse_map[x] for x in G.nodes])
    new_G.add_edges_from(new_edge_list)
    return new_G, index_map


def unindex_graph(G: nx.Graph, index_map):
    nodes = list(G.nodes)
    old_edges = []
    for a, b in G.edges:
        old_edges.append((index_map[a], index_map[b]))
    old_G = nx.Graph()
    old_G.add_edges_from(old_edges)
    return old_G


def pythagorean_fn(p, distance_dict):
    num_params = len(p)
    param_index_combinations = distance_dict.keys()
    equations = []
    for i, j in param_index_combinations:
        equations.append(p[i]**2 + p[j]**2 - distance_dict[(i, j)]**2)
    return equations


def get_intra_cluster_distances(connection_matrix):
    num_clusters = connection_matrix.shape[0]
    cluster_combinations = combinations(range(num_clusters), 2)
    dist_dict = {}
    for combo in cluster_combinations:
        dist_dict[combo] = connection_matrix[combo[0], combo[1]]
    pyth_partial_fn = partial(pythagorean_fn, distance_dict=dist_dict)
    # solutions = fsolve(pyth_partial_fn, [1]*num_clusters)
    solutions = least_squares(pyth_partial_fn, [1] * num_clusters)
    return solutions.x


def label_to_onehot(labels):
    uniq_labels = list(set(labels))
    num_labels = len(uniq_labels)
    num_samples = len(labels)
    oh_labels = []
    label_to_oh = {}
    for i, l in enumerate(uniq_labels):
        tmp = np.zeros(num_labels)
        tmp[i] = 1
        label_to_oh[l] = tmp
    for label in labels:
        oh_labels.append(label_to_oh[label])
    return np.vstack(oh_labels)


def get_uniform_random_sbm(community_cap: int, members_cap: int) -> (nx.Graph, np.ndarray):
    """
    Creates an SBM graph from a uniform random distribution
    :param community_cap: The number of maximum communities in the graph
    :param members_cap: The maximum number of individuals in a community
    :return: A networkx Graph and the labels for the nodes
    """
    num_communities = random.randint(2, community_cap)
    vertices_per_community = []
    for i in range(num_communities):
        num_community_vertices = random.randint(2, members_cap)
        vertices_per_community.append(num_community_vertices)
    p = np.zeros((num_communities, num_communities))

    for combo in combinations(range(num_communities), 2):
        community_edge_prob = random.random() * 0 # TODO make this higher
        p[combo[0], combo[1]] = community_edge_prob
        p[combo[1], combo[0]] = community_edge_prob

    for i in range(num_communities):
        community_edge_prob = random.random() * 0.1
        p[i, i] = min(1., community_edge_prob + 0.2 + random.random() * 0.5)

    adj, labels = sbm(vertices_per_community, p, return_labels=True)
    G = nx.convert_matrix.from_numpy_array(adj)
    return G, labels


def get_distance_matrix(G: nx.Graph, labels):
    num_edges_dict = defaultdict(int)
    node_to_label = {}
    communities = sorted(list(set(labels)))
    community_to_index = dict(zip(communities, range(len(communities))))
    for i, n in enumerate(G.nodes):
        node_to_label[n] = labels[i]
    for u, v in G.edges:
        c1 = community_to_index[node_to_label[u]]
        c2 = community_to_index[node_to_label[v]]
        if c1 == c2:
            continue
        if c1 > c2:
            c1, c2 = c2, c1
        num_edges_dict[(c1, c2)] += 1
    num_communities = len(communities)
    d = np.zeros((num_communities, num_communities))
    for i in range(num_communities):
        for j in range(num_communities):
            x, y = (j, i) if i > j else (i, j)
            d[i, j] = num_edges_dict[(x, y)]
    return d, opposite_dict(community_to_index)


def get_community_embeddings_from_gt(G, labels):
    # oh_embeddings = label_to_onehot(labels)
    d_matrix, index_to_community = get_distance_matrix(G, labels)
    inv_d_matrix = np.max(d_matrix) + 1 - d_matrix
    community_vec_magnitudes = get_intra_cluster_distances(inv_d_matrix)
    c_embeddings = label_to_onehot(labels) * community_vec_magnitudes
    return c_embeddings


def get_reduced_community_embeddings_from_gt(G, labels, dim=2, reduction_fn=TSNE):
    c_embeddings = get_community_embeddings_from_gt(G, labels)
    reducer = reduction_fn(dim)
    embeddings = reducer.fit_transform(c_embeddings)
    return embeddings


def perturb_embeddings(embedding: np.ndarray, intensity: float):
    num_samples, dim = embedding.shape
    new_embedding = np.zeros_like(embedding)
    total_energy = 0
    for i, emb in enumerate(embedding):
        perturbation_vector = np.random.rand(dim)
        perturbation_vector = perturbation_vector / np.linalg.norm(perturbation_vector)
        p_magnitude = random.random() * intensity
        perturbation = perturbation_vector * p_magnitude
        total_energy += p_magnitude
        new_embedding[i] = emb + perturbation
    return new_embedding, total_energy







