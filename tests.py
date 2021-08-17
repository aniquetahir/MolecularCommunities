import unittest
from experiment_md import evaluate_embedding
import numpy as np
from scipy import sparse

class EvaluationTests(unittest.TestCase):
    def test_execution(self):
        num_nodes = 1000
        dim = 3
        num_classes = 20
        embedding = np.random.rand(num_nodes, dim)
        labels = np.round(np.random.rand(num_nodes, num_classes))
        sparse_labels = sparse.csc_matrix(labels)
        evaluate_embedding(embedding, sparse_labels)