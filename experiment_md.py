from scipy.io import loadmat
import networkx as nx
from typing import Dict
from sklearn.manifold import MDS
from sklearn.linear_model import LogisticRegression
from itertools import combinations, product

def evaluate_embedding(embedding, labels):
    # Apply log regression
    # Accuracy, F1, precision, recall
    # 10 fold validation
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
    param_whatever = {
        'a': range(2, 5),
        'b': range(20, 50, 10)
    }
    x = iter_params(param_whatever)
    bc_mat = loadmat('/new-pool/datasets/blogcatalog.mat')
    G = nx.from_scipy_sparse_matrix(bc_mat['network'])
    labels = bc_mat['group']
    # Get md embedding

    # Apply Log regression
    # Accuracy, F1, precision, recall
    # 10 fold X validation