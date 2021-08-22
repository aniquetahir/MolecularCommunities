import pickle
from collections import defaultdict
import numpy as np
import networkx as nx
from typing import Dict


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
