import random
from tqdm import tqdm

from anique import *

if __name__ == "__main__":
    num_samples = 10000
    synthetic_data = []
    for _ in tqdm(range(num_samples)):
        G, labels = get_uniform_random_sbm(50, 200)
        # 2-d embeddings
        r_embeddings = get_reduced_community_embeddings_from_gt(G, labels)
        # Get energy after perturbation
        for perturb_intensity in range(0, 1000, 100):
            pert_embeddings, energy = perturb_embeddings(r_embeddings, perturb_intensity)
            synthetic_data.append((pert_embeddings, energy,))

    # TODO save synthetic data
    save_pickle(synthetic_data, 'synthetic_energy_data.pkl')