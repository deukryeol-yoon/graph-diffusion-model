import networkx as nx
import numpy as np
import torch


def get_dataset(dataset="grid"):
    if dataset == "grid":
        graphs = []
        for i in range(2, 5):
            for j in range(2, 5):
                graphs.append(nx.grid_2d_graph(i, j))

        return graphs
    else:
        raise NotImplementedError("")


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs: list, max_num_nodes: int, features="id"):

        self.max_num_nodes = max_num_nodes
        self.adj = []
        self.features = []
        self.lens = []

        for g in graphs:
            adj_ = nx.to_numpy_matrix(g)
            self.adj.append(np.asarray(adj_) + np.identity(g.number_of_nodes()))

        if features == "id":
            self.features.append(np.identity(max_num_nodes))

    def __len__(self):
        return len(self.adj)

    def __getitem__(self, idx):

        adj = self.adj[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        adj_decoded = np.zeros(self.max_num_nodes * (self.max_num_nodes + 1) // 2)
        node_idx = 0

        adj_vectorized = adj_padded[
            np.triu(np.ones((self.max_num_nodes, self.max_num_nodes))) == 1
        ]

        features = self.features[idx]

        return {"adj": adj_padded, "adj_decoded": adj_vectorized, "features": features}
