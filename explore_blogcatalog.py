from scipy.io import loadmat
import networkx as nx
from main import generate_embeddings_n2v


def mat_to_n2v_emb(mat_path, save_path):
    blogcatalog = loadmat(mat_path)
    network = blogcatalog['network']
    communities = blogcatalog['group']
    print('Creating networkx graph...')
    G = nx.from_scipy_sparse_matrix(network)
    print('Generating graph embeddings...')
    model = generate_embeddings_n2v(G)
    model.wv.save_word2vec_format(save_path)


if __name__ == '__main__':
    mat_to_n2v_emb('/dmml_pool/datasets/graph/youtube.mat', 'n2v_youtube.emb')
    mat_to_n2v_emb('/dmml_pool/datasets/graph/flickr.mat', 'n2v_flickr.emb')
    mat_to_n2v_emb('/dmml_pool/datasets/graph/youtube.mat', 'n2v_youtube.emb')
    pass
