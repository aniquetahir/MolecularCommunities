from tqdm import tqdm

def read_communities(filepath: str):
    communities = []
    print('Reading community file')
    with open(filepath) as community_file:
        for line in tqdm(community_file, total=5000):
            cty = line.split('\t')
            cty = [int(x) for x in cty]
            communities.append(cty)
    return communities


if __name__ == '__main__':
    communities = read_communities('datasets/livejournal/com-lj.top5000.cmty.txt')
    nodes = set([x for y in communities for x in y])
    #

    pass