import math
import os
import unittest

from sklearn import metrics
from scipy.io import loadmat
import networkx as nx
from typing import Dict
from sklearn.manifold import MDS
from sklearn.linear_model import LogisticRegression
from itertools import combinations, product
from test_cmd import get_multiembeddings, get_cumulative_embeddings
import numpy as np
# from experiment import n2v_embeddings
import pickle
import random
from scipy import sparse


# class TestPartitions(unittest.TestCase):
#     def test_partitions(self):
#         a = [1, 2, 3, 4]
#         p = get_partitions(a, 2)
#         self.assertEqual(p[0], [1, 2])

def get_partitions(arr, num_partitions):
    partitions = []
    size_fold = math.floor(len(arr)/num_partitions)
    for i in range(0, len(arr), size_fold):
        partitions.append(arr[i:i+size_fold])
    return partitions


def get_metrics(predictions, labels) -> (float, float, float, float):
    """

    :param predictions: prediction probabilities
    :param labels: one hot labels
    :return: accuracy, precision, recall, f1
    """
    # TODO complete this function
    pass


def evaluate_embedding(embedding, oh_labels, folds=10, sparse=True, average='micro'):
    embedding_and_labels = list(zip(embedding, np.argmax(oh_labels, axis=1) if sparse else oh_labels))
    random.shuffle(embedding_and_labels)
    partitions = get_partitions(embedding_and_labels, folds)
    predictions = []
    for i in range(folds):
        test_fold = partitions[i]
        training_folds = [y for j, x in enumerate(partitions) if j != i for y in x]
        test_embeddings = np.vstack([x[0] for x in test_fold])
        test_labels = np.array(np.hstack([x[1] for x in test_fold])).flatten()
        train_embeddings = np.vstack([x[0] for x in training_folds])
        train_labels = np.array(np.hstack([x[1] for x in training_folds])).flatten()
        # Apply log regression
        log_model = LogisticRegression()
        log_model.fit(train_embeddings, train_labels)
        fold_preds = log_model.predict(test_embeddings)
        predictions.append((test_labels, fold_preds))
    # Accuracy, F1, precision, recall
    test_label_sets = np.hstack([x[0] for x in predictions])
    test_prediction_sets = np.hstack([x[1] for x in predictions])
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(test_label_sets, test_prediction_sets, average=average)
    accuracy = metrics.accuracy_score(test_label_sets, test_prediction_sets)
    # 10 fold validation
    return accuracy, precision, recall, fscore, support


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


def save_md_embedding_combo(G, g_name='bc'):
    randomization_nums = [100, 200, 400]
    skim_nums = [10, 20, 40, 80]
    dims = [2, 3]
    params = {
        'G': [G],
        'num_embeddings': randomization_nums,
        'skim': skim_nums,
        'dim': dims,
        'g_name': [g_name]
    }
    func_params = iter_params(params)
    num_combos = len(func_params)
    for i, combo in enumerate(func_params):
        print(f'Combo {i}/{num_combos}')
        savename = f'md_{g_name}_dim{combo["dim"]}_{combo["num_embeddings"]}_s{combo["skim"]}.pkl'
        if not os.path.exists(savename):
            m_embeddings = get_multiembeddings(**combo)
            with open(savename, 'wb') as embedding_file:
                pickle.dump(m_embeddings, embedding_file)
        else:
            with open(savename, 'rb') as embedding_file:
                m_embeddings = pickle.load(embedding_file)
            mds = MDS(2, n_jobs=35)
            reduced_embedding = mds.fit_transform(m_embeddings)
            with open(savename + '.mds2d', 'wb') as mds_file:
                pickle.dump(reduced_embedding, mds_file)
            




if __name__ == "__main__":
    bc_mat = loadmat('/dmml_pool/datasets/graph/blogcatalog.mat')
    G = nx.from_scipy_sparse_matrix(bc_mat['network'])
    labels = bc_mat['group']
    save_md_embedding_combo(G)
    print('Done saving combos')
    # Get md embedding
    num_randomizations = 200
    skim = 50
    reduced_dim = 3
    print('Calculating Molecular embeddings')
    m_embeddings = get_multiembeddings(G, num_randomizations, skim=skim)
    with open(f'md_bc_{num_randomizations}_s{skim}.pkl', 'wb') as md_file:
        pickle.dump(m_embeddings, md_file)
    print('Calculating MDS reduction')
    mds = MDS(reduced_dim, n_jobs=3)
    reduced_embedding = mds.fit_transform(m_embeddings)
    # Evaluate embedding
    with open(f'md_bc_mds{reduced_dim}_{num_randomizations}_s{skim}.pkl', 'wb') as embedding_file:
        pickle.dump(reduced_embedding, embedding_file)
    print('Calculating Metrics')
    acc_md, prec_md, recall_md, f1_md, support = evaluate_embedding(reduced_embedding, labels)
    print(f'Acc: {acc_md}\n Prec: {prec_md}\n Recall: {recall_md}\n F1: {f1_md}')

    # Get n2v embedding
    # n2v_emb = n2v_embeddings('/dmml_pool/datasets/graph/blogcatalog.mat', 'sc_bc_walks.pkl', 'sc_bc.emb', emb_dim=3)
    # reduced_embedding = mds.fit(np.array(n2v_emb))
    # acc_n2v, prec_n2v, recall_n2v, f1_n2v = evaluate_embedding(reduced_embedding, labels)
