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
        hk.Linear(128), jax.nn.leaky_relu,
        hk.Linear(64), jax.nn.leaky_relu,
        hk.Linear(DIM)
    ])
    return net(batch)




def train():
    dim = DIM
    num_fire_steps = 100
    key = jax.random.PRNGKey(7)
    key, split = jax.random.split(key)
    net = hk.without_apply_rng(hk.transform(net_fn))
    params = net.init(split, np.ones((100, dim)))

    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(params)

    print('Hyperparameter shape:')
    print(jax.tree_map(lambda x: x.shape, params))
    graph_generator = generate_sample(10, 100)
    dt_start = 0.001
    dt_max = 0.004
    num_iterations = 1000
    displacement, shift = space.free()


    @jit
    def loss_fn(params, x, y, bonds, energy):
        def nn_fn(dr):
            return net.apply(params, dr)
        energy_fn = smap.bond(nn_fn, displacement, bonds)
        init, apply = minimize.fire_descent(energy_fn, shift, dt_start=dt_start, dt_max=dt_max)
        # init(np.ones((100, 2)))
        # apply = jit(apply)
        # @jit
        def scan_fn(state, i):
            return apply(state), 0.
        state = init(np.array(x, dtype=f64))
        # apply(state)
        state, _ = lax.scan(scan_fn, state, np.arange(num_fire_steps))
        num_samples = x.shape[0]
        return np.sum(np.square(state.position - y))/(energy + 1)

    loss_fn_value_grad = jax.value_and_grad(loss_fn)

    for i in tqdm(range(num_iterations)):
        if i % 10 == 1:
            save_pickle(params, f'training_cache/simulated_nn.{i}.pkl')

        try:
            edges, perterbed_emb, energy, gt_embeddings = next(graph_generator)
            # loss_fn(params, np.array(perterbed_emb, f64), np.array(gt_embeddings, f64), np.array(edges))
            loss_value, grads = loss_fn_value_grad(params, np.array(perterbed_emb, f64), np.array(gt_embeddings, dtype=f64),
                                                   np.array(edges), energy)
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
    pass