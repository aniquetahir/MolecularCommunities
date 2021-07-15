# This is a sample Python script.
import snap
import os
import pandas as pd
from tqdm import tqdm
import networkx as nx
# import node2vec.src.node2vec as node2vec
from node2vec import Node2Vec

from gensim.models import Word2Vec
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

N2V_P = 1
N2V_Q = 1
N2V_NUM_WALKS = 10
N2V_WALK_LENGTH = 80
N2V_DIM = 128
N2V_WINDOW_SIZE = 10
N2V_ITER = 1
N2V_WORKERS = 8


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def read_ungraph(filepath: str):
    print('Reading edges file')
    table = pd.read_csv(filepath, sep='\t', comment='#', names=['from', 'to'])
    return table

def read_communities(filepath: str):
    communities = []
    print('Reading community file')
    with open(filepath) as community_file:
        for line in tqdm(community_file, total=5000):
            cty = line.split('\t')
            cty = [int(x) for x in cty]
            communities.append(cty)
    return communities
            
def create_n2v_embeddings(graph_location, save_path):
    #communities = read_communities(cmty_location)
    edges = read_ungraph(graph_location)
    G = nx.Graph()
    print('Adding edges to networkx graph')
    G.add_edges_from(edges.to_numpy(), nodetype=int)
    # Set weights to 1 because of horrible coding in Stanfords node2vec library
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    print(G.number_of_nodes())
    print('Creating n2v graph')
    n2v = Node2Vec(G, dimensions=N2V_DIM, walk_length=N2V_WALK_LENGTH, num_walks=N2V_NUM_WALKS, workers=N2V_WORKERS)
    print('Embedding nodes')
    model = n2v.fit(window=N2V_WINDOW_SIZE, min_count=0, batch_words=4)
    print('Saving model')
    model.wv.save_word2vec_format(save_path)
    


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_n2v_embeddings('datasets/livejournal/com-lj.ungraph.txt', 'lj.n2v.emb')
    create_n2v_embeddings('datasets/friendster/com-friendster.ungraph.txt', 'friendster.n2v.emb')
    create_n2v_embeddings('datasets/orkut/com-orkut.ungraph.txt', 'orkut.n2v.emb')
    # communities = read_communities('datasets/livejournal/com-lj.top5000.cmty.txt')
    # edges = read_ungraph('datasets/livejournal/com-lj.ungraph.txt')
    # G = nx.Graph()
    # print('Adding edges to networkx graph')
    # G.add_edges_from(edges.to_numpy()[:1000], nodetype=int)
    # # Set weights to 1 because of horrible coding in Stanfords node2vec library
    # for edge in G.edges():
    #     G[edge[0]][edge[1]]['weight'] = 1
    # print(G.number_of_nodes())
    # print('Creating n2v graph')
    # n2v = Node2Vec(G, dimensions=N2V_DIM, walk_length=N2V_WALK_LENGTH, num_walks=N2V_NUM_WALKS, workers=N2V_WORKERS)
    # print('Embedding nodes')
    # model = n2v.fit(window=N2V_WINDOW_SIZE, min_count=0, batch_words=4)
    # print('Saving model')
    # model.wv.save_word2vec_format('n2v.emb')
