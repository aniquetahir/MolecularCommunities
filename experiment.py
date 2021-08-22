import stellargraph as sg
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import datasets
import stellargraph as sg
from stellargraph.data import BiasedRandomWalk
from IPython.display import display, HTML
from scipy.io import loadmat
import ge

import pickle
from gensim.models import Word2Vec, KeyedVectors

BC_DATA_PATH = '/new-pool/datasets/blogcatalog.mat'
FLICKR_DATA_PATH = ''
YOUTUBE_DATA_PATH = ''

def n2v_embedding_nx(G: nx.Graph, path_to_walks, emb_path, emb_dim, p=0.5, q=2.0):
    bc_G = G.copy()
    sg_bc_G = sg.StellarGraph.from_networkx(bc_G)
    print(sg_bc_G.info())
    print('='*10)
    rw = BiasedRandomWalk(sg_bc_G)

    if os.path.exists(path_to_walks):
        with open(path_to_walks, 'rb') as walks_file:
            walks = pickle.load(walks_file)
    else:
        walks = rw.run(
            nodes=list(sg_bc_G.nodes()),  # root nodes
            length=100,  # maximum length of a random walk
            n=10,  # number of random walks per root node
            p=p,  # Defines (unormalised) probability, 1/p, of returning to source node
            q=q,  # Defines (unormalised) probability, 1/q, for moving away from source node
        )
        print("Number of random walks: {}".format(len(walks)))
        with open(path_to_walks, 'wb') as walk_file:
            pickle.dump(walks, walk_file)
    print("Done walking.")
    str_walks = [[str(n) for n in walk] for walk in walks]
    full_emb_path = f'{emb_path}_{emb_dim}'
    if not os.path.exists(full_emb_path):
        model = Word2Vec(str_walks, vector_size=emb_dim, window=5, min_count=0, sg=1, workers=2, epochs=1)
        model.wv.save(full_emb_path)
        model = model.wv
    else:
        model = KeyedVectors.load(fname=full_emb_path)
    return model


def n2v_embeddings(path_to_mat, path_to_walks, emb_path, emb_dim=128):
    bc_mat = loadmat(path_to_mat)
    bc_G = nx.from_scipy_sparse_matrix(bc_mat['network'])
    sg_bc_G = sg.StellarGraph.from_networkx(bc_G)
    print(sg_bc_G.info())
    print('='*10)
    rw = BiasedRandomWalk(sg_bc_G)

    if os.path.exists(path_to_walks):
        with open(path_to_walks, 'rb') as walks_file:
            walks = pickle.load(walks_file)
    else:
        walks = rw.run(
            nodes=list(sg_bc_G.nodes()),  # root nodes
            length=100,  # maximum length of a random walk
            n=10,  # number of random walks per root node
            p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
            q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
        )
        print("Number of random walks: {}".format(len(walks)))
        with open(path_to_walks, 'wb') as walk_file:
            pickle.dump(walks, walk_file)
    print("Done walking.")
    str_walks = [[str(n) for n in walk] for walk in walks]
    full_emb_path = f'{emb_path}_{emb_dim}'
    if not os.path.exists(full_emb_path):
        model = Word2Vec(str_walks, vector_size=emb_dim, window=5, min_count=0, sg=1, workers=2, epochs=1)
        model.wv.save_word2vec_format(full_emb_path)
    return model


def generic_embeddings(embedding_function, function_params, train_params, path_to_mat, emb_path):
    if not os.path.exists(emb_path):
        mat = loadmat(path_to_mat)
        G = nx.from_scipy_sparse_matrix(mat['network'])
        function_params['graph'] = G
        model = embedding_function(**function_params)
        model.train(**train_params)
        embeddings = model.get_embeddings()
        with open(emb_path, 'wb') as emb_file:
            pickle.dump(embeddings, emb_file)


if __name__ == "__main__":
    # print('=' * 20)
    # print('Blog Catalog')
    # n2v_embeddings('/dmml_pool/datasets/graph/blogcatalog.mat', 'sc_bc_walks.pkl', 'sc_bc.emb')
    # print('=' * 20)
    # print('Flickr')
    # n2v_embeddings('/dmml_pool/datasets/graph/flickr.mat', 'sc_flickr_walks.pkl', 'sc_flickr.emb')
    # print('=' * 20)
    # print('YouTube')
    # n2v_embeddings('/dmml_pool/datasets/graph/youtube.mat', 'sc_youtube_walks.pkl', 'sc_flickr.emb')


    # DeepWalk
    dw_params = {
        'walk_length': 10,
        'num_walks': 80,
        'workers': 4
    }
    dw_train_params = {
        'window_size': 5,
        'iter': 3
    }
    generic_embeddings(ge.DeepWalk, dw_params, dw_train_params, BC_DATA_PATH, 'bc_deepwalk.emb')

    # LINE
    line_params = {
        'embedding_size': 128,
        'order': 'second'
    }
    line_train_params = {
        'batch_size': 1024,
        'epochs': 50,
        'verbose': 2
    }
    generic_embeddings(ge.LINE, line_params, line_train_params, BC_DATA_PATH, 'bc_line_128.emb')





