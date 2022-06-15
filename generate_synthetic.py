import networkx as nx
import os
import pickle
import numpy as np
import dgl
from node2vec import Node2Vec
import os
from tqdm import tqdm
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

def make_grid_graphs(lower=10, upper=20, **kwargs):
    graphs = []
    for i in range(lower, upper):
        for j in range(lower, upper):
            graphs.append(nx.grid_2d_graph(i, j))

    return [dgl.DGLGraph(g).to_networkx() for g in graphs]

def make_lobster_graphs(seed=1234, mean_num_nodes=80, num_graphs=100, **kwargs):
    path = 'data/lobster.h5'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    # Same as GRAN
    graphs = []
    p1 = 0.7
    p2 = 0.7
    min_node = 10
    max_node = 100
    tmp_seed = seed

    while len(graphs) < num_graphs:
        g = nx.random_lobster(mean_num_nodes, p1, p2, seed=tmp_seed)
        if g.number_of_nodes() >= min_node and g.number_of_nodes() <= max_node:
            graphs.append(g)
        tmp_seed += 1
    pickle.dump(graphs, open(path, 'wb'))
    return graphs
    # return [ for _ in range(num_graphs)]

def make_community_graph(c_sizes, g_p=0.3, p_inter=0.05):#, p_inter=0.01):
    graphs = [nx.fast_gnp_random_graph(c_size, g_p, seed=np.random.choice(1000)) for c_size in c_sizes]
    G = nx.disjoint_union_all(graphs)
    communities = [G.subgraph(c) for c in nx.connected_components(G)]
    for ix, community1 in enumerate(communities):
        nodes1 = list(community1.nodes())
        for community2 in communities[ix + 1:]:
            nodes2 = list(community2.nodes())

            num_inter_edges = int((len(nodes1) + len(nodes2)) * p_inter)
            # print(num_inter_edges, community1.number_of_edges(), community2.number_of_edges())
            edges1 = np.random.choice(nodes1, size=num_inter_edges)
            edges2 = np.random.choice(nodes2, size=num_inter_edges)
            G.add_edges_from(zip(edges1, edges2))
    return G

def make_community_graphs():
    path = 'data/community.h5'
    #if os.path.exists(path):
    #    return pickle.load(open(path, 'rb'))

    graphs = []
    num_communities = 2
    for k in range(1000):
        c_sizes = [np.random.choice(list(range(1,8)))]   ### 원래 30, 80
        graphs.append(make_community_graph(c_sizes * num_communities))

    #pickle.dump(graphs, open(path, 'wb'))
    return graphs

def make_community_graphs_large():
    path = 'data/community_large.h5'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))

    graphs = []
    num_communities = 10
    for k in range(400):
        c_sizes = np.random.choice(list(range(100, 200)), size=num_communities)
        graphs.append(make_community_graph(c_sizes, g_p=0.15, p_inter=0.025))

    pickle.dump(graphs, open(path, 'wb'))
    return graphs

def make_ego_graphs():
    import scipy.sparse as sp
    def parse_index_file(filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def Graph_load(dataset = 'cora'):
        '''
        Load a single graph dataset
        :param dataset: dataset name
        :return:
        '''
        names = ['x', 'tx', 'allx', 'graph']
        objects = []

        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/{}/ind.{}.test.index".format(dataset, dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        G = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(G)
        return adj, features, G

    #_, _, G = Graph_load(dataset='citeseer')
    dataset = Planetoid(name = 'pubmed', root = './dataset')
    G = to_networkx(dataset[0]).to_undirected()
    

    subgraphs = [G.subgraph(c) for c in nx.connected_components(G)]
    G = max(subgraphs, key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=3)
        if G_ego.number_of_nodes() >= 1 and (G_ego.number_of_nodes() <= 20):
            graphs.append(G_ego)
    return graphs

def make_er_graphs(seed, p, lower=5, upper=20, **kwargs):
    if kwargs.get('num_graphs') is None: # Generate based on the lower/upper range
        graphs = []
        for num_nodes in range(lower, upper):
            for i in range(35):
              graphs.append(nx.erdos_renyi_graph(num_nodes, p, seed=seed+2*i))
              graphs.append(nx.erdos_renyi_graph(num_nodes, p, seed=seed + 2*i+1))
        return graphs
    elif kwargs.get('num_graphs') is not None: # Generate based on num_graphs and given size
        return [nx.erdos_renyi_graph(kwargs.get('mean_num_nodes'), p, seed=seed) for _ in range(kwargs.get('num_graphs'))]

def load_graphs(name, min_num_nodes=20,
                  max_num_nodes=500,
                  node_attributes=False,
                  graph_labels=False):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    data_dir = 'data'
    G = nx.Graph()
    # load data
    path = os.path.join(data_dir, name)
    data_adj = np.loadtxt(
        os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
    if node_attributes:
      data_node_att = np.loadtxt(
          os.path.join(path, '{}_node_attributes.txt'.format(name)),
          delimiter=',')
    data_node_label = np.loadtxt(
        os.path.join(path, '{}_node_labels.txt'.format(name)),
        delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(
        os.path.join(path, '{}_graph_indicator.txt'.format(name)),
        delimiter=',').astype(int)
    if graph_labels:
      data_graph_labels = np.loadtxt(
          os.path.join(path, '{}_graph_labels.txt'.format(name)),
          delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
      if node_attributes:
        G.add_node(i + 1, feature=data_node_att[i])
      G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # remove self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
      # find the nodes for each graph
      nodes = node_list[data_graph_indicator == i + 1]
      G_sub = G.subgraph(nodes)
      if graph_labels:
        G_sub.graph['label'] = data_graph_labels[i]
      # print('nodes', G_sub.number_of_nodes())
      # print('edges', G_sub.number_of_edges())
      # print('label', G_sub.graph)
      if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
      ) <= max_num_nodes:
        graphs.append(G_sub)
        if G_sub.number_of_nodes() > max_nodes:
          max_nodes = G_sub.number_of_nodes()
        # print(G_sub.number_of_nodes(), 'i', i)
        # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
        # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    return graphs

def load_proteins(min_num_nodes=100,
                  max_num_nodes=500,
                  node_attributes=False,
                  graph_labels=False):
    return load_graphs('DD', min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
            node_attributes=node_attributes, graph_labels=graph_labels)

def load_firstmm_db(min_num_nodes=0,
        max_num_nodes=10000,
        node_attributes=False,
        graph_labels=False):
    return load_graphs('FIRSTMM_DB', min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
            node_attributes=node_attributes, graph_labels=graph_labels)


def load_zinc():
    return pickle.load(open('data/zinc.h5', 'rb'))[: 1000]

import torch_geometric
from torch_geometric.utils.convert import from_networkx

def save_graph(graphs, folder_name, num_node_features = 16):
    directory = './data/' + folder_name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, graph in tqdm(enumerate(graphs)):
        geometric_graph = from_networkx(graph)
        n2v = Node2Vec(graph = graph,
                        dimensions = num_node_features,
                        walk_length = 2,
                        p = 1,
                        q = 0.0001,
                        weight_key = None,
                        num_walks = 100,
                        workers = 1)
        model = n2v.fit(window=10, min_count=1)
        geometric_graph.node_features = model.wv
        torch.save(geometric_graph, directory + "graph-"+str(i)+'.graph')
        
if __name__ == "__main__":
    # generate synthetic graph
    graphs = make_grid_graphs()
    
    # generate node features using node2vec and save graph.
    save_graph(graphs, 'grid')

    # load graph (sample)
    graph = torch.load('./data/grid/graph-0.graph')
    print(graph.node_features.get_vector('0'))