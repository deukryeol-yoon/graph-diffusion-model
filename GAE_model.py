import torch
import torch.nn as nn
import torch.nn.init as init

import scipy.optimize

import GAE_util


device = "cuda" if torch.cuda.is_available() else "cpu"


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim)).to(device)

    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        return y


class GraphVAE(nn.Module):
    def __init__(self, encoder, decoder, embed_dim, max_num_nodes):

        super(GraphVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.embed_dim = embed_dim
        self.max_num_nodes = max_num_nodes

    def edge_similarity_matrix(
        self, adj, adj_recon, matching_features, matching_features_recon, sim_func
    ):
        S = torch.zeros(
            self.max_num_nodes,
            self.max_num_nodes,
            self.max_num_nodes,
            self.max_num_nodes,
        )
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, j, a, a] = (
                            adj[i, j]
                            * adj_recon[a, a]
                            * sim_func(matching_features[i], matching_features_recon[a])
                        )
                        # with feature not implemented
                        # if input_features is not None:
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = (
                                adj[i, j]
                                * adj[i, i]
                                * adj[j, j]
                                * adj_recon[a, b]
                                * adj_recon[a, a]
                                * adj_recon[b, b]
                            )
        return S

    def deg_feature_similarity(self, f1, f2):
        return 1 / (abs(f1 - f2) + 1)

    def mpm(self, x_init, S, max_iters=50):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [
                        torch.max(x[j, :] * S[i, j, a, :])
                        for j in range(self.max_num_nodes)
                        if j != i
                    ]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            norm = torch.norm(x_new)
            x = x_new / norm
        return x

    def generate(self):

        """return only adjacency matrix for now."""

        z = torch.autograd.Variable(torch.rand(self.embed_dim)).to(device)
        out = self.decoder(z)

        recon_adj_lower = util.recover_adj_lower(out.cpu().data, self.max_num_nodes)
        recon_adj_tensor = util.recover_full_adj_from_lower(recon_adj_lower)

        return recon_adj_tensor

    def forward(self, x, adj):

        mu, sigma = self.encoder(x, adj)

        sigma2 = sigma.mul(0.5).exp_()

        eps = torch.autograd.Variable(torch.randn(sigma.size())).to(device)
        z = mu     #####
        out = self.decoder(z)

        recon_adj_lower = util.recover_adj_lower(out.cpu().data, self.max_num_nodes)
        recon_adj_tensor = util.recover_full_adj_from_lower(recon_adj_lower)

        out_features = torch.sum(recon_adj_tensor, 1)
        adj_features = torch.sum(adj, 1).cpu().data[0]

        S = self.edge_similarity_matrix(
            adj.cpu().data[0],
            recon_adj_tensor,
            adj_features.cpu().data,
            out_features.cpu().data,
            self.deg_feature_similarity,
        )

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        assignment = self.mpm(init_assignment, S)

        # matching
        # use negative of the assignment score since the alg finds min cost flow
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())

        # order row index according to col index
        # adj_permuted = self.permute_adj(adj_data, row_ind, col_ind)
        adj_permuted = adj.cpu().data[0]
        adj_vectorized = adj_permuted[
            torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1
        ].squeeze_()
        adj_vectorized_var = torch.autograd.Variable(
            adj_vectorized, requires_grad=False
        ).to(device)

        adj_recon_loss = util.adj_recon_loss(out[0], adj_vectorized_var)
        loss_kl = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes  # normalize

        loss = adj_recon_loss    #####

        return loss


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=embed_dim)

        self.linear_mu = nn.Linear(embed_dim, embed_dim)
        self.linear_sigma = nn.Linear(embed_dim, embed_dim)

        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.relu(x)
        x = self.conv2(x, adj)

        # Using this to aggregate node info to graph info
        x = torch.sum(x, 1)

        mu = self.linear_mu(x)
        sigma = self.linear_sigma(x)

        return mu, sigma


class GraphDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, max_num_nodes):
        super(GraphDecoder, self).__init__()

        output_dim = max_num_nodes * (max_num_nodes + 1) // 2

        self.conv1 = nn.Linear(embed_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)

        self.max_num_nodes = max_num_nodes

        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return torch.sigmoid(x)
