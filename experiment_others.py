from experiment_md import iter_params
from experiment import n2v_embedding_nx
import networkx as nx




def evaluate_karate():
    G = nx.karate_club_graph()
    labels = [v['club'] for k, v in G.nodes.data()]
    label_to_cat = {}
    for i, v in enumerate(set(labels)):
        label_to_cat[v] = i
    labels = [label_to_cat[x] for x in labels]
    model = n2v_embedding_nx(G, 'n2v_karate.walks', 'n2v_karate.emb', 2)
    print(model.wv)

    pass


if __name__ == "__main__":
    evaluate_karate()
    pass
