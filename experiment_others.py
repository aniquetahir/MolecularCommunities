import numpy as np
import pandas as pd

from experiment_md import iter_params
from experiment import n2v_embedding_nx
from experiment_md import evaluate_embedding
import networkx as nx
from scipy.io import loadmat


def evaluate_blogcatalog_n2v():
    bc_mat = loadmat('blogcatalog.mat')
    G = nx.from_scipy_sparse_matrix(bc_mat['network'])
    labels = bc_mat['group']
    evaluate_generic_n2v(G, labels, 'bc', 'n2v_bc.walks', 'n2v_bc.emb', True)


def evaluate_generic_n2v(G, labels, name, walks_path, embedding_path, sparse_labels=False):
    def get_model(G, p, q, dim):
        return n2v_embedding_nx(G, walks_path, embedding_path, emb_dim=dim, p=p, q=q)
    params_combinations = {
        'G': [G],
        'p': [0.25, 0.5, 0.75, 1],
        'q': [0.25, 0.5, 0.75, 1],
        'dim': [2, 4, 8, 16, 32, 64]
    }
    combos = iter_params(params_combinations)
    eval_results = []
    for combo in combos:
        eval_object = combo
        model = get_model(**combo)
        # print(model.vectors)
        # Get back original node configuration
        embedding = []
        for n in G.nodes:
            embedding.append(model.vectors[model.key_to_index[str(n)]])
        embeddings = np.vstack(embedding)
        acc, prec, recall, f1, support = evaluate_embedding(embeddings, labels, sparse=sparse_labels)
        del eval_object['G'], model, embeddings
        eval_object['accuracy'] = acc
        eval_object['precision'] = prec
        eval_object['recall'] = recall
        eval_object['fscore'] = f1
        eval_results.append(eval_object)
    results_frame = pd.DataFrame(eval_results)
    results_frame.to_csv(f'n2v_{name}_evaluation.csv')


def evaluate_karate_n2v():
    G = nx.karate_club_graph()
    labels = [v['club'] for k, v in G.nodes.data()]
    label_to_cat = {}
    for i, v in enumerate(set(labels)):
        label_to_cat[v] = i
    labels = [label_to_cat[x] for x in labels]
    evaluate_generic_n2v(G, labels, 'karate', 'n2v_karate.walks', 'n2v_karate.emb')


if __name__ == "__main__":
    # evaluate_karate_n2v()
    evaluate_blogcatalog_n2v()
    pass
