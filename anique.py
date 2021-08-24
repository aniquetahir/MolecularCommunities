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
