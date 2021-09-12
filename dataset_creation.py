import random

import graspologic.simulations
from tqdm import tqdm
import jax
import networkx as nx
from anique import *
import numpy as np
import jax.numpy as jnp


def generate_dcsbm_samples(max_communities=50, max_community_members=200):
    key = jax.random.PRNGKey(9)
    while True:
        num_communities = np.random.randint(2, max_communities)
        n = [int(x) for x in np.round(np.random.uniform(low=2, high=max_community_members, size=num_communities))]

        max_edge_prob = 0.1
        min_intra_edge_prob = max_edge_prob + 0.1
        max_intra_edge_prob = min(1, min_intra_edge_prob + 0.5)

        inter_community_probs = np.random.uniform(high=max_edge_prob, size=(num_communities, num_communities))
        tri = np.tril(inter_community_probs, -1)
        inter_community_probs = tri + tri.T + \
            np.diag([np.random.uniform(min_intra_edge_prob, max_intra_edge_prob) for _ in range(num_communities)])
        # p = inter_community_probs
        adj_matrix, labels = graspologic.simulations.sbm(n, inter_community_probs, return_labels=True,
                                                         dc=lambda: 1 - np.random.randint(1, 100))

        G = nx.from_numpy_array(adj_matrix)
        key, split = jax.random.split(key)
        num_nodes = G.number_of_nodes()
        random_embeddings = jax.random.uniform(split, (num_nodes, 2), maxval=jnp.sqrt(num_nodes))
        yield list(G.edges), random_embeddings, labels, num_nodes



def generate_training_samples(max_communities=50, max_community_members=200):
    key = jax.random.PRNGKey(7)
    while True:
        G, labels = get_uniform_random_sbm(max_communities, max_community_members)
        key, split = jax.random.split(key)
        r_embeddings = jax.random.uniform(key, (G.number_of_nodes(), 2), maxval=10.)
        yield list(G.edges), r_embeddings, labels


def generate_sample_with_labels(max_communities=50, max_community_members=200):
    synthetic_data = []
    while True:
        G, labels = get_uniform_random_sbm(max_communities, max_community_members)
        # 2-d embeddings
        r_embeddings = get_reduced_community_embeddings_from_gt(G, labels)
        # Get energy after perturbation
        perturb_intensity = random.choice([0, 1000, 100])
        pert_embeddings, energy = perturb_embeddings(r_embeddings, perturb_intensity)
        yield list(G.edges), pert_embeddings, energy, r_embeddings, labels
        del G, labels, pert_embeddings, r_embeddings


def generate_sample(max_communities=50, max_community_members=200):
    synthetic_data = []
    while True:
        G, labels = get_uniform_random_sbm(max_communities, max_community_members)
        # 2-d embeddings
        r_embeddings = get_reduced_community_embeddings_from_gt(G, labels)
        # Get energy after perturbation
        perturb_intensity = random.choice([0, 1000, 100])
        pert_embeddings, energy = perturb_embeddings(r_embeddings, perturb_intensity)
        yield list(G.edges), pert_embeddings, energy, r_embeddings
        del G, labels, pert_embeddings, r_embeddings


if __name__ == "__main__":
    num_samples = 10000
    synthetic_data = []
    for i in tqdm(range(num_samples)):
        if i % 10 == 1:
            save_pickle(synthetic_data, f'synth_cache/data.{i}.pkl')
            del synthetic_data
            synthetic_data = []
        G, labels = get_uniform_random_sbm(50, 200)
        # 2-d embeddings
        r_embeddings = get_reduced_community_embeddings_from_gt(G, labels)
        # Get energy after perturbation
        for perturb_intensity in range(0, 1000, 100):
            pert_embeddings, energy = perturb_embeddings(r_embeddings, perturb_intensity)
            synthetic_data.append((list(G.edges), pert_embeddings, energy,))
        del G, labels

    # TODO save synthetic data
    save_pickle(synthetic_data, 'synthetic_energy_data.pkl')