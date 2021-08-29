from typing import Sequence
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
import haiku as hk
from jax import vmap, pmap, jit
import os
from anique import load_pickle
import random


def get_batch():
    key = jax.random.PRNGKey(8)
    while True:
        key, split = jax.random.split(key)
        x = jax.random.uniform(split, (100, 1), dtype=jnp.float32) * 10
        y = x ** 2
        yield x, y

def get_graph_batch():
    random.seed(7)
    graph_files = [x for x in os.listdir('synth_cache') if 'pkl' in x]
    while True:
        filename = random.choice(graph_files)
        graph_array = load_pickle(filename)
        graph_choice = random.choice(graph_array)
        # Create x, y from chosen graph
        edges, embeddings, energy = graph_choice
        num_nodes = embeddings.shape[0]
        avg_energy = energy/num_nodes
        x = []
        y = []
        for u, v in edges:
            dr = jnp.linalg.norm(embeddings[u]-embeddings[v])
            x.append(dr)
        x = jnp.array(x, dtype=jnp.float32).reshape((len(x), 1))
        y = jnp.ones_like(x, dtype=jnp.float32) * avg_energy
        yield x, y


def net_fn(batch):
    mlp = hk.Sequential([
        hk.Linear(128), jax.nn.leaky_relu,
        hk.Linear(64), jax.nn.leaky_relu,
        hk.Linear(1)
    ])
    return mlp(batch)


def training_loop(generator, num_iterations=1000):
    key, split = jax.random.split(jax.random.PRNGKey(7))
    net = hk.without_apply_rng(hk.transform(net_fn))
    init_x, init_y = next(generator)
    params = net.init(split, init_x)
    key, split = jax.random.split(key)

    @jit
    def mse(params, x, y_batch):
        y_pred = net.apply(params, x)
        num_samples = y_pred.shape[0]
        return jnp.sum(jnp.square(y_pred-y_batch))/num_samples

    loss_grad_fn = jax.grad(mse)
    optimizer = optax.adamw(1e-2)
    opt_state = optimizer.init(params)

    for i in tqdm(range(num_iterations)):
        x, y = next(generator)
        if i % 100 == 0:
            print(mse(params, x, y))
            pass
        grads = loss_grad_fn(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return net, params


if __name__ == "__main__":
    # jax.config.update('jax_platform_name', 'cpu')
    gen = get_batch()
    net, params = training_loop(gen)
    print('done')
