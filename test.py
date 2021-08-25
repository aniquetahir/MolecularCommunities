from collections import defaultdict
from anique import *
import numpy as np
import networkx as nx
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA
from scipy.io import loadmat


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


if __name__ == "__main__":
    # See blogcatalog
    bc_mat = loadmat('blogcatalog.mat')
    G = nx.from_scipy_sparse_matrix(bc_mat['network'])

    labels = onehot_to_cat(bc_mat['group'])
    print('Embeddings are one hot labels')
    embeddings = np.array(bc_mat['group'].todense())
    G, embeddings, labels, _ = get_reduced_sample_graph(G, embeddings, labels, 1000)
    d_matrix, index_to_community = get_distance_matrix(G, labels)
    # reverse distances
    inv_d_matrix = np.max(d_matrix) + 1 - d_matrix
    community_vec_magnitudes = get_intra_cluster_distances(inv_d_matrix)

    print('Got labels')
    # print('PCA')
    # plot_reduced_embedding(G, embeddings, labels, PCA)
    # print('Truncated SVD')
    # plot_reduced_embedding(G, embeddings, labels, TruncatedSVD)
    # print('Kernet PCA')
    # plot_reduced_embedding(G, embeddings, labels, KernelPCA)
    # print('MDS')
    # plot_reduced_embedding(G, embeddings, labels, MDS)
    # print('Isomap')
    # plot_reduced_embedding(G, embeddings, labels, Isomap)
    print('TSNE')
    plot_reduced_embedding(G, embeddings, labels, TSNE)
