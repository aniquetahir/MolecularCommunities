import math
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
from experiment import n2v_embeddings
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


if __name__ == "__main__":
    bc_mat = loadmat('/dmml_pool/datasets/graph/blogcatalog.mat')
    G = nx.from_scipy_sparse_matrix(bc_mat['network'])
    labels = bc_mat['group']
    # Get md embedding
    num_randomizations = 200
    skim = 50
    m_embeddings = get_multiembeddings(G, num_randomizations, skim=skim)
    mds = MDS(3, n_jobs=3)
    reduced_embedding = mds.fit_transform(m_embeddings)
    # Evaluate embedding
    with open(f'md_bc_{num_randomizations}_s{skim}.pkl', 'wb') as embedding_file:
        pickle.dump(reduced_embedding, embedding_file)

    acc_md, prec_md, recall_md, f1_md, support = evaluate_embedding(reduced_embedding, labels)
    print(f'Acc: {acc_md}\n Prec: {prec_md}\n Recall: {recall_md}\n F1: {f1_md}')

    # Get n2v embedding
    # n2v_emb = n2v_embeddings('/dmml_pool/datasets/graph/blogcatalog.mat', 'sc_bc_walks.pkl', 'sc_bc.emb', emb_dim=3)
    # reduced_embedding = mds.fit(np.array(n2v_emb))
    # acc_n2v, prec_n2v, recall_n2v, f1_n2v = evaluate_embedding(reduced_embedding, labels)
