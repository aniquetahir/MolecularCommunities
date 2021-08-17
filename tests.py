import unittest
from experiment_md import evaluate_embedding
import numpy as np
from scipy import sparse
from sklearn.datasets import load_iris


class EvaluationTests(unittest.TestCase):
    def test_execution(self):
        embedding, sparse_labels = load_iris(return_X_y=True)
        evaluate_embedding(embedding, sparse_labels, sparse=False)