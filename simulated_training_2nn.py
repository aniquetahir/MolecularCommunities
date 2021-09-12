from anique import save_pickle, load_pickle
import jax.nn
import numpy as onp

import haiku as hk
import optax

from anique import *
import os

import jax.numpy as np
from jax import random
from jax import jit, grad, vmap, value_and_grad
from jax import lax, partial
from jax import ops

from jax.config import config
config.update("jax_enable_x64", True)

from jax_md import space, smap, energy, minimize, quantity, simulate, partition

from functools import partial
import time

f32 = np.float32
f64 = np.float64

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

DIM = 2
from dataset_creation import generate_sample_with_labels, generate_training_samples, generate_dcsbm_samples


@jit
def elite_rescale(x: np.ndarray) -> np.ndarray:
    a = x - np.min(x, axis=0)
    a = a/np.max(a, axis=0)
    return np.nan_to_num(a)


def dropout_fn(x):
    return hk.dropout(hk.next_rng_key(), 0.5, x)


def net_fn(batch):
    net = hk.Sequential([
        hk.Linear(256, name='n1_l1'), jax.nn.leaky_relu, dropout_fn,
        hk.Linear(128, name='n1_l2'), jax.nn.leaky_relu, dropout_fn,
        hk.Linear(DIM, name='n1_l3')
    ])
    return net(batch)


def net2_fn(batch):
    net = hk.Sequential([
        hk.Linear(256, name='n2_l1'), jax.nn.leaky_relu, dropout_fn,
        hk.Linear(128, name='n2_l2'), jax.nn.leaky_relu, dropout_fn,
        hk.Linear(DIM, name='n2_l3')
    ])
    return net(batch)


def community_coexistence_matrix(labels):
    num_samples = len(labels)
    coexistence_matrix = onp.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            if labels[i] == labels[j]:
                coexistence_matrix[i, j] = 1
    return coexistence_matrix


def train():
    dim = DIM
    num_fire_steps = 100
    key = jax.random.PRNGKey(7)
    key, split = jax.random.split(key)
    net = hk.transform(net_fn)
    net2 = hk.transform(net2_fn)

    params = None
    if os.path.exists('2nn_dim2_params.pkl'):
        params1 = net.init(split, np.ones((100, dim)))
        key, split = jax.random.split(key)
        params2 = net2.init(split, np.ones((100, dim)))
        params = hk.data_structures.merge(params1, params2)
    else:
        params = load_pickle('2nn_dim2_params.pkl')

    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(params)
    # opt_state2 = optimizer.init(params2)

    print('Hyperparameter shape:')
    print(jax.tree_map(lambda x: x.shape, params))
    graph_generator = generate_dcsbm_samples(5, 100) # generate_training_samples(5, 100)  # generate_sample_with_labels(5, 100)
    dt_start = 0.001
    dt_max = 0.004
    num_iterations = 1000


    loss_history = []
    for i in tqdm(range(num_iterations)):
        if i % 10 == 1:
            save_pickle(params, f'training_cache/simulated_nn_combined.{i}.pkl')
            mean_loss = np.mean(np.hstack(loss_history[-10:]))
            print(f'AVERAGE LOSS: {mean_loss}')
            del mean_loss

        try:
            # edges, perterbed_emb, energy, gt_embeddings, labels = next(graph_generator)
            # loss_fn(params, np.array(perterbed_emb, f64), np.array(gt_embeddings, f64), np.array(edges))
            edges, perturbed_emb, labels, num_nodes = next(graph_generator)

            displacement, shift = space.periodic(num_nodes ** (1./DIM), wrapped=False) # space.free()

            @jit
            def loss_metric(embeddings, coexistence_matrix):
                product_map = space.map_product(displacement)

                def sec_norm(R):
                    pmap = product_map(R, R)
                    pmap = np.where(pmap == 0, 0.0001, pmap)
                    return np.linalg.norm(pmap, axis=2)

                all_distances = np.abs(sec_norm(embeddings))
                num_intra = np.sum(coexistence_matrix)
                num_inter = np.sum(1 - coexistence_matrix)

                intra_community_distances = all_distances * coexistence_matrix
                inter_community_distances = all_distances * (1 - coexistence_matrix)
                mean_intra = np.sum(intra_community_distances) / num_intra
                mean_inter = np.sum(inter_community_distances) / num_inter
                return mean_intra - mean_inter

            @jit
            def loss_fn(params, x, bonds, key, community_matrix):
                key, split = jax.random.split(key)
                # num_points = x.shape[0]

                def bond_nn_fn(dr):
                    return net.apply(params, split, dr)

                def common_nn_fn(dr):
                    return net2.apply(params, split, dr)

                bond_energy_fn = smap.bond(bond_nn_fn, displacement, bonds)
                common_energy_fn = smap.pair(common_nn_fn, displacement)

                def combined_energy_fn(R):
                    return bond_energy_fn(R) + common_energy_fn(R)

                init, apply = minimize.fire_descent(combined_energy_fn, shift, dt_start=dt_start, dt_max=dt_max)
                # init(np.ones((100, 2)))
                # apply = jit(apply)
                # @jit
                def rescale(x):
                    t = x - np.min(x, axis=0)
                    return t/np.linalg.norm(t)

                def scan_fn(state, i):
                    return apply(state), 0.

                state = init(np.array(x, dtype=f64))
                # apply(state)
                state, _ = lax.scan(scan_fn, state, np.arange(num_fire_steps))
                # num_samples = x.shape[0]
                # return (np.sum(np.square(elite_rescale(state.position) - elite_rescale(y))))/num_samples
                return loss_metric(state.position, community_matrix)

            loss_fn_value_grad = jax.value_and_grad(loss_fn)


            key, split = jax.random.split(key)
            cm = community_coexistence_matrix(labels)
            loss_value, grads = loss_fn_value_grad(params, np.array(perturbed_emb, f64),
                                                   np.array(edges), split, cm)
            loss_history.append(loss_value)
            print(f"Loss: {loss_value}")
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            del grads, edges, perturbed_emb, updates
        except Exception as e:
            print(e)
            input()
            print(perturbed_emb)
            input()
            # print(gt_embeddings)

    return params


if __name__ == "__main__":
    params = train()
    save_pickle(params, '2nn_dim2_params.pkl')
    pass