import unittest
from experiment_md import evaluate_embedding
import numpy as np
from scipy import sparse
import jax
from jax import jit, numpy as jnp
from jax_md.space import map_product
from jax_md import space
from sklearn.datasets import load_iris


class EvaluationTests(unittest.TestCase):
    def test_execution(self):
        embedding, sparse_labels = load_iris(return_X_y=True)
        evaluate_embedding(embedding, sparse_labels, sparse=False)

class EvaluateJaxFns(unittest.TestCase):

    def test_metric(self):
        displacement, shift = space.free()
        product_map = map_product(displacement)
        key = jax.random.PRNGKey(9)
        key, split = jax.random.split(key)
        test_embedding = jax.random.uniform(split, (100, 2)) * 10
        key, split = jax.random.split(key)
        coexistence_matrix = jax.random.uniform(split, (100, 100))
        coexistence_matrix = jnp.round(coexistence_matrix)

        @jit
        def metric_fn(embeddings, coexistence_matrix):
            def sec_norm(R):
                pmap = product_map(R, R)
                pmap = jnp.where(pmap == 0, 0.0001, pmap)
                return jnp.linalg.norm(pmap, axis=2)

            all_distances = jnp.abs(sec_norm(embeddings))
            num_intra = jnp.sum(coexistence_matrix)
            num_inter = jnp.sum(1 - coexistence_matrix)

            intra_community_distances = all_distances * coexistence_matrix
            inter_community_distances = all_distances * (1 - coexistence_matrix)
            mean_intra = jnp.sum(intra_community_distances) / num_intra
            mean_inter = jnp.sum(inter_community_distances) / num_inter
            return mean_intra - mean_inter

        metric_fn_grad = jax.grad(metric_fn)
        grads = metric_fn_grad(test_embedding, coexistence_matrix)

        assert not jnp.isnan(grads).any()
