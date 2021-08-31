import random
from tqdm import tqdm

from anique import *

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