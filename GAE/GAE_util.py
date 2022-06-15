import torch
import torch.nn.functional as F


def recover_adj_lower(vec, max_num_nodes):
    # NOTE: Assumes 1 per minibatch
    adj = torch.zeros(max_num_nodes, max_num_nodes)
    adj[torch.triu(torch.ones(max_num_nodes, max_num_nodes)) == 1] = vec
    return adj


def recover_full_adj_from_lower(lower):
    diag = torch.diag(torch.diag(lower, 0))
    return lower + torch.transpose(lower, 0, 1) - diag


def adj_recon_loss(adj_truth, adj_pred):
    return F.binary_cross_entropy(adj_truth, adj_pred)
