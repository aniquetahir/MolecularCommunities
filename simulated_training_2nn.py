from anique import save_pickle, load_pickle
import jax.nn
import numpy as onp

import haiku as hk
import optax

from anique import *

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
from dataset_creation import generate_sample

def net_fn(batch):
    net = hk.Sequential([
        hk.Linear(256, name='n1_l1'), jax.nn.leaky_relu,
        hk.Linear(128, name='n1_l2'), jax.nn.leaky_relu,
        hk.Linear(DIM, name='n1_l3')
    ])
    return net(batch)

def net2_fn(batch):
    net = hk.Sequential([
        hk.Linear(256, name='n2_l1'), jax.nn.leaky_relu,
        hk.Linear(128, name='n2_l2'), jax.nn.leaky_relu,
        hk.Linear(DIM, name='n2_l3')
    ])
    return net(batch)



def train():
    dim = DIM
    num_fire_steps = 100
    key = jax.random.PRNGKey(7)
    key, split = jax.random.split(key)
    net = hk.without_apply_rng(hk.transform(net_fn))
    net2 = hk.without_apply_rng(hk.transform(net2_fn))

    params1 = net.init(split, np.ones((100, dim)))
    key, split = jax.random.split(key)
    params2 = net2.init(split, np.ones((100, dim)))
    params = hk.data_structures.merge(params1, params2)

    optimizer = optax.adamw(1e-5)
    opt_state = optimizer.init(params)
    # opt_state2 = optimizer.init(params2)

    print('Hyperparameter shape:')
    print(jax.tree_map(lambda x: x.shape, params))
    graph_generator = generate_sample(5, 100)
    dt_start = 0.001
    dt_max = 0.004
    num_iterations = 1000
    displacement, shift = space.free()


    @jit
    def loss_fn(params, x, y, bonds, energy):
        # num_points = x.shape[0]
        def bond_nn_fn(dr):
            return net.apply(params, dr)

        def common_nn_fn(dr):
            return net2.apply(params, dr)

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
        num_samples = x.shape[0]
        return (np.sum(np.square(rescale(state.position) - rescale(y))))/num_samples

    loss_fn_value_grad = jax.value_and_grad(loss_fn)

    loss_history = []
    for i in tqdm(range(num_iterations)):
        if i % 10 == 1:
            save_pickle(params, f'training_cache/simulated_nn_combined.{i}.pkl')
            mean_loss = np.mean(np.hstack(loss_history[-10:]))
            print(f'AVERAGE LOSS: {mean_loss}')
            del mean_loss

        try:
            edges, perterbed_emb, energy, gt_embeddings = next(graph_generator)
            # loss_fn(params, np.array(perterbed_emb, f64), np.array(gt_embeddings, f64), np.array(edges))
            loss_value, grads = loss_fn_value_grad(params, np.array(perterbed_emb, f64), np.array(gt_embeddings, dtype=f64),
                                                   np.array(edges), energy)
            loss_history.append(loss_value)
            print(f"Loss: {loss_value}")
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            del grads, edges, perterbed_emb, energy, gt_embeddings, updates
        except Exception as e:
            print(e)
            input()
            print(perterbed_emb)
            input()
            print(gt_embeddings)

    return params


if __name__ == "__main__":
    params = train()
    save_pickle(params, '2nn_dim2_params.pkl')
    pass