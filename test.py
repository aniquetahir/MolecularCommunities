from anique import *
import numpy as np
import networkx as nx
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA


from scipy.io import loadmat

if __name__ == "__main__":
    # See blogcatalog
    bc_mat = loadmat('blogcatalog.mat')
    G = nx.from_scipy_sparse_matrix(bc_mat['network'])

    labels = onehot_to_cat(bc_mat['group'])
    print('Embeddings are one hot labels')
    embeddings = np.array(bc_mat['group'].todense())
    G, embeddings, labels, _ = get_reduced_sample_graph(G, embeddings, labels, 1000)
    plot_reduced_embedding(G, embeddings, labels, PCA)
    plot_reduced_embedding(G, embeddings, labels, TruncatedSVD)
    plot_reduced_embedding(G, embeddings, labels, KernelPCA)
    plot_reduced_embedding(G, embeddings, labels, MDS)
    plot_reduced_embedding(G, embeddings, labels, Isomap)
    plot_reduced_embedding(G, embeddings, labels, TSNE)


    pass