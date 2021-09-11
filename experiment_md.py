import math

import jax.random
import pandas as pd
import os
import unittest
from tqdm import tqdm
from sklearn import metrics
from scipy.io import loadmat
import networkx as nx
from typing import Dict
from sklearn.manifold import MDS
from sklearn.linear_model import LogisticRegression
from community_md import FullNNMolecularCommunities
from itertools import combinations, product
from test_cmd import get_multiembeddings, get_cumulative_embeddings
import numpy as np
# from experiment import n2v_embeddings
import pickle
import random
from test_cmd import plot_graph
from scipy import sparse
from anique import *


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


def get_subset(G: nx.Graph, labels, num_samples, sparse=False):
    nodes = list(G.nodes)
    if num_samples > G.number_of_nodes():
        return G, labels
    m_labels = labels
    if sparse:
        m_labels = onehot_to_cat(m_labels)
    removal_nodes = random.sample(nodes, max(0, G.number_of_nodes() - num_samples))
    new_G = G.copy(G)
    new_G.remove_nodes_from(removal_nodes)
    new_labels = []

    for i in nodes:
        if i not in removal_nodes:
            new_labels.append(m_labels[i])

    new_G, _ = readjust_graph(new_G)
    return new_G, new_labels


def evaluate_bc_nn_sample(sample_size=1000, num_runs=10):
    label_stats = defaultdict(int)
    bc_mat = loadmat('blogcatalog.mat')
    G = nx.from_scipy_sparse_matrix(bc_mat['network'])
    labels = bc_mat['group']
    G, labels = get_subset(G, labels, sample_size, sparse=True)
    for label in labels:
        label_stats[label] += 1
    print(label_stats)
    random.seed(5)
    key = jax.random.PRNGKey(5)
    for i in range(num_runs):
        key, split = jax.random.split(key)
        model = FullNNMolecularCommunities(split, G, minimization_steps=1000)
        embeddings, energy = model.train()
        acc, pre, recall, fscore, support = evaluate_embedding(embeddings, labels)
        print('=' * 20)
        print(f'Acc: {acc}\nPre: {pre}\nRecall: {recall}\nF-score: {fscore}')
        print('=' * 20)


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


def save_md_embedding_combo(G, labels, g_name='bc', sparse_labels=False):
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
    eval_results = []
    for i, combo in enumerate(func_params):
        print(f'Combo {i}/{num_combos}')
        savename = f'md_{g_name}_dim{combo["dim"]}_{combo["num_embeddings"]}_s{combo["skim"]}.pkl'
        m_embeddings = None
        reduced_embedding = None
        if not os.path.exists(savename):
            m_embeddings = get_multiembeddings(**combo)
            with open(savename, 'wb') as embedding_file:
                pickle.dump(m_embeddings, embedding_file)
        else:
            with open(savename, 'rb') as embedding_file:
                m_embeddings = pickle.load(embedding_file)

        if np.isnan(m_embeddings).any():
            print(f'nan found at:')
            print(combo)
        for d in [2, 4, 8, 16, 32, 64]:
            mds = MDS(d, n_jobs=40)
            mds_path = savename + f'.mds{d}d'
            if not os.path.exists(mds_path):
                reduced_embedding = mds.fit_transform(m_embeddings)
                with open(mds_path, 'wb') as mds_file:
                    pickle.dump(reduced_embedding, mds_file)
            else:
                with open(mds_path, 'rb') as mds_file:
                    reduced_embedding = pickle.load(mds_file)
            acc_md, prec_md, recall_md, f1_md, support = evaluate_embedding(reduced_embedding, labels, sparse=sparse_labels)
            print(f'MDS dim:{d}')
            print(combo)
            print(f'Acc: {acc_md}\n Prec: {prec_md}\n Recall: {recall_md}\n F1: {f1_md}')
            eval_object = {
                'mds_dim': d,
                'accuracy': acc_md,
                'precision': prec_md,
                'recall': recall_md,
                'f1': f1_md
            }
            for k, v in combo.items():
                eval_object[k] = v
            eval_results.append(eval_object)
    results_frame = pd.DataFrame(eval_results)
    results_frame.to_csv(f'{g_name}.results.csv')


def evaluate_md_karate():
    G = nx.karate_club_graph()
    labels = [v['club'] for k, v in G.nodes.data()]
    label_to_cat = {}
    for i, v in enumerate(set(labels)):
        label_to_cat[v] = i
    labels = [label_to_cat[x] for x in labels]
    save_md_embedding_combo(G, labels, g_name='karate')
    print('Done saving combos')


from collections import  defaultdict


def evaluate_md_karate_nn():
    G = nx.karate_club_graph()
    labels = [v['club'] for k, v in G.nodes.data()]
    label_to_cat = {}
    for i, v in tqdm(enumerate(set(labels))):
        label_to_cat[v] = i
    labels = [label_to_cat[x] for x in labels]
    gt_communties = defaultdict(list)
    for i, label in enumerate(labels):
        gt_communties[label].append(i)
    gt_communties = list(gt_communties.values())

    key = jax.random.PRNGKey(7)
    for i in range(10):
        key, split = jax.random.split(key)
        md_system = FullNNMolecularCommunities(split, G, minimization_steps=1000)
        fullnn_embeddings, max_energy = md_system.train()
        # save_md_embedding_combo(G, labels, g_name='karate_fullnn')
        acc, precision, recall, f1, support = evaluate_embedding(fullnn_embeddings, labels, sparse=False)
        plot_graph(G, fullnn_embeddings, gt_communties)
        print('=' * 20)
        print(f'Acc: {acc}\n Precision: {precision}\n Recall: {recall}\n F1: {f1}')
        print('=' * 20)
    # Completed combos
    print('Done saving combos')


def evaluate_md_blogcatalog():
    bc_mat = loadmat('blogcatalog.mat')
    G = nx.from_scipy_sparse_matrix(bc_mat['network'])
    labels = bc_mat['group']
    save_md_embedding_combo(G, labels, g_name='bc', sparse_labels=True)


if __name__ == "__main__":
    evaluate_bc_nn_sample()
    # evaluate_md_karate_nn()
    # evaluate_md_blogcatalog()
    # evaluate_md_blogcatalog()
    # bc_mat = loadmat('/dmml_pool/datasets/graph/blogcatalog.mat')
    # G = nx.from_scipy_sparse_matrix(bc_mat['network'])
    # labels = bc_mat['group']
    # Get n2v embedding
    # n2v_emb = n2v_embeddings('/dmml_pool/datasets/graph/blogcatalog.mat', 'sc_bc_walks.pkl', 'sc_bc.emb', emb_dim=3)
    # reduced_embedding = mds.fit(np.array(n2v_emb))
    # acc_n2v, prec_n2v, recall_n2v, f1_n2v = evaluate_embedding(reduced_embedding, labels)
