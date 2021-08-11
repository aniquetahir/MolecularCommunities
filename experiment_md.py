from scipy.io import loadmat
import networkx as nx
from typing import Dict
from sklearn.manifold import MDS
from sklearn.linear_model import LogisticRegression
from itertools import combinations, product
from test_cmd import get_multiembeddings, get_cumulative_embeddings
import numpy as np
from experiment import n2v_embeddings
import pickle


def evaluate_embedding(embedding, oh_labels):
    # Apply log regression
    # Accuracy, F1, precision, recall
    # 10 fold validation
    return embedding, oh_labels
    pass


def iter_params(params: Dict):
    param_lists = []
    keys = []
    for k, v in params.items():
        param_lists.append(v)
        keys.append(k)
    comb = product(*param_lists)
    dicted_combinations = []
    for x in comb:
        t_param_dict = {}
        for i, y in enumerate(x):
            t_param_dict[keys[i]] = y
        dicted_combinations.append(t_param_dict)
    return dicted_combinations


if __name__ == "__main__":
    bc_mat = loadmat('/new-pool/datasets/blogcatalog.mat')
    G = nx.from_scipy_sparse_matrix(bc_mat['network'])
    labels = bc_mat['group']
    # Get md embedding
    m_embeddings = get_multiembeddings(G, 200, skim=30)
    mds = MDS(3)
    reduced_embedding = mds.fit(m_embeddings)
    # Evaluate embedding
    with open('md_bc_200_s30.pkl', 'wb') as embedding_file:
        pickle.dump(reduced_embedding, embedding_file)
    acc_md, prec_md, recall_md, f1_md = evaluate_embedding(reduced_embedding, labels)

    # Get n2v embedding
    n2v_emb = n2v_embeddings('/dmml_pool/datasets/graph/blogcatalog.mat', 'sc_bc_walks.pkl', 'sc_bc.emb', emb_dim=3)
    reduced_embedding = mds.fit(np.array(n2v_emb))
    acc_n2v, prec_n2v, recall_n2v, f1_n2v = evaluate_embedding(reduced_embedding, labels)





