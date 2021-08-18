import pickle

from community_md import MolecularCommunities
from collections import defaultdict
from sklearn.manifold import MDS
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from jax import random
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def get_cumulative_embeddings(G, num_embeddings, num_steps=10000, skim=-1, dim=2):
    all_embeddings = []
    all_energies = []
    cumulative_embedding = None
    for i in tqdm(range(num_embeddings)):
        key = random.PRNGKey(i)
        mc = MolecularCommunities(key, G, minimization_steps=num_steps, pos=cumulative_embedding, dim=dim)
        md_embeddings, energy = mc.train()
        all_embeddings.append(md_embeddings)
        all_energies.append(energy)
        skimmed_embeddings = sorted(zip(all_embeddings, all_energies), key=lambda x: x[1])
        all_embeddings = [x[0] for x in skimmed_embeddings]
        if skim == -1:
            tmp_embeddings = jnp.hstack(all_embeddings)
        else:
            tmp_embeddings = jnp.hstack(all_embeddings[:skim])
        mds = MDS(dim)
        cumulative_embedding = mds.fit_transform(tmp_embeddings)
        del mc
        del tmp_embeddings
        del skimmed_embeddings
    return cumulative_embedding, all_embeddings


def get_multiembeddings(G, num_embeddings, num_steps=10000, skim=-1, dim=2, g_name=None):
    all_embeddings = []
    all_energies = []
    graph_hash = g_name
    for i in tqdm(range(num_embeddings)):
        md_hash = f'{graph_hash}_n{num_steps}_d{dim}_{i}'
        hash_path = f'md_cache/{md_hash}'
        if not os.path.exists(hash_path) or g_name is None:
            key = random.PRNGKey(i)
            mc = MolecularCommunities(key, G, minimization_steps=num_steps, dim=dim)
            md_embeddings, energy = mc.train()
            md_embeddings = np.array(md_embeddings)
            del mc
            with open(hash_path, 'wb') as md_file:
                pickle.dump((md_embeddings, energy), md_file)
        else:
            with open(hash_path, 'rb') as md_file:
                md_embeddings, energy = pickle.load(md_file)
        all_embeddings.append(md_embeddings)
        all_energies.append(energy)
        # del md_embeddings
        # del mc
    skimmed_embeddings = sorted(zip(all_embeddings, all_energies), key=lambda x: x[1])
    all_embeddings = [x[0] for x in skimmed_embeddings]
    all_embeddings = np.hstack(all_embeddings) if skim == -1 else np.hstack(all_embeddings[:skim])
    return all_embeddings


def get_minembeddings(G, num_embeddings, num_steps=10000):
    min_embedding = None
    min_energy = float('inf')
    for i in range(num_embeddings):
        key = random.PRNGKey(i)
        mc = MolecularCommunities(key, G, minimization_steps=num_steps)
        mds_embeddings, energy = mc.train()
        if energy < min_energy:
            min_embedding = mds_embeddings
        del mc
    return min_embedding


def plot_graph(embeddings, communities):
    pos_md = dict(zip(range(len(embeddings)), np.array(embeddings)))
    nx.draw_networkx_nodes(G, pos_md, nodelist=communities[0], node_color='r')
    nx.draw_networkx_nodes(G, pos_md, nodelist=communities[1], node_color='g')
    if len(communities)>2:
        nx.draw_networkx_nodes(G, pos_md, nodelist=communities[2], node_color='b')
    nx.draw_networkx_edges(G, pos_md)
    plt.show()


if __name__ == "__main__":
    G = nx.karate_club_graph()
    communities = greedy_modularity_communities(G)
    gt_communities = defaultdict(list)
    for k, v in G.nodes.data():
        gt_communities[v['club']].append(k)

    gt_communities = list(gt_communities.values())
    multiembeddings = get_multiembeddings(G, 100, skim=50)

    mds = MDS(2)
    mds_embeddings = mds.fit_transform(np.array(multiembeddings))
    # mds_embeddings = get_minembeddings(G, 100)
    plot_graph(mds_embeddings, gt_communities)
    plot_graph(mds_embeddings, communities)

    print('Testing Community MD')

