from GAE_data import get_dataset, GraphDataset
from GAE_model import GraphVAE, GraphEncoder, GraphDecoder
from generate_synthetic import make_community_graph, make_community_graphs, make_ego_graphs, make_er_graphs

import argparse

import os
import dgl
from node2vec import Node2Vec
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import QM7b
from torch_geometric.datasets import QM9
from torch_geometric.datasets import ZINC
from torch_geometric.datasets import TUDataset
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
import pickle


device = "cuda" if torch.cuda.is_available() else "cpu"
LR_milestones = [500, 1000]


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")

    parser.add_argument("--lr", dest="lr", type=float, help="Learning rate.")
    parser.add_argument("--batch_size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument(
        "--max_num_nodes",
        dest="max_num_nodes",
        type=int,
        help="Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.",
    )
    parser.add_argument(
        "--feature",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )

    parser.set_defaults(
        dataset="grid",
        feature_type="id",
        lr=0.001,
        batch_size=1,
        num_workers=1,
        max_num_nodes=-1,
    )
    parser.add_argument("-f")
    return parser.parse_args()


def build_model(args, hidden_dim=64, embed_dim=256):
    out_dim = args.max_num_nodes * (args.max_num_nodes + 1) // 2
    if args.feature_type == "id":
        input_dim = args.max_num_nodes
    elif args.feature_type == "deg":
        input_dim = 1
    elif args.feature_type == "struct":
        input_dim = 2

    encoder = GraphEncoder(input_dim, hidden_dim, embed_dim).to(device)
    decoder = GraphDecoder(embed_dim, hidden_dim, args.max_num_nodes).to(device)

    return GraphVAE(encoder, decoder, embed_dim, args.max_num_nodes).to(device)


def train(model, dataloader, args, epochs=5000):

    model.train()

    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)
    
    graph_embeddings = []

    for epoch in range(epochs):
        for batch_idx, data in enumerate(dataloader):
            model.zero_grad()
            features = data["features"].float()
            adj_input = data["adj"].float()

            features = torch.autograd.Variable(features).to(device)
            adj_input = torch.autograd.Variable(adj_input).to(device)

            loss = model(features, adj_input)
            print("Epoch: ", epoch, ", Iter: ", batch_idx, ", Loss: ", loss.item())
            loss.backward()

            optimizer.step()
            scheduler.step()

            if batch_idx == 1000 :
              break
              
        model.eval()
        for batch_idx, data in enumerate(dataloader):
            
            features = data["features"].float()
            adj_input = data["adj"].float()
            features = torch.autograd.Variable(features).to(device)
            adj_input = torch.autograd.Variable(adj_input).to(device)
            
            mu_,sigma_ = model.encoder(features, adj_input)
            graph_embeddings.append(mu_)
            
          
    with open('graph_embeddings_synthetic_er.pkl', 'wb') as f:
      pickle.dump(graph_embeddings, f)
    files.download('graph_embeddings_synthetic_er.pkl')


def generate(model):

    adj_recon = model.generate()
    nx.draw(nx.from_numpy_matrix(np.rint(adj_recon.cpu().numpy())))
    plt.show()


def main():

    args = arg_parse()
    graphs = make_er_graphs(1,0.5)
    if args.max_num_nodes == -1:
        args.max_num_nodes = max(
            [graphs[i].number_of_nodes() for i in range(len(graphs))]
        )
    else:
        max_num_nodes = args.max_num_nodes
        # remove graphs with number of nodes greater than max_num_nodes
        graphs = [g for g in graphs if g.number_of_nodes() <= max_num_nodes]

    dataset = GraphDataset(graphs, args.max_num_nodes, args.feature_type)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size
    )

    model = build_model(args)
    train(model, dataloader, args, epochs=1)

    with open('graphs_synthetic_er.pkl', 'wb') as f:
      pickle.dump(graphs,f)
    files.download('graphs_synthetic_er.pkl')

    with open('model_synthetic_er.pkl', 'wb') as f:
      pickle.dump(model, f)
    files.download('model_synthetic_er.pkl')
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size
    )
  


if __name__ == "__main__":
    main()
