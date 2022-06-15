import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.lsun import LSUN
from torch.utils.data import Subset
import numpy as np

from torch.utils.data import Dataset
import pickle
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import QM7b
from torch_geometric.datasets import QM9
from torch_geometric.datasets import ZINC
from torch_geometric.datasets import TUDataset


# Our DataSet ----------------------------------------------------------------------------------
class GraphEmbeddings(Dataset):

    def __init__(self, name, root_dir, model_path, embedding_path, image_size=16, features="id", transform=None, device=None, direct_embedding_flag=False):
        self.root_dir = root_dir
        self.device = device
        self.name = name
        self.image_size = image_size
        self.direct_embedding = direct_embedding_flag
        if self.direct_embedding:
            graphs = None
            with open(embedding_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            if torch.is_tensor(self.embeddings) is False:
                for emb in self.embeddings:
                    emb.requires_grad = False
                self.embeddings = torch.stack(self.embeddings)
            else:
                self.embeddings = self.embeddings.detach()
            if len(self.embeddings.shape) == 3:
                self.embeddings = self.embeddings.squeeze(1)
            print(self.embeddings.shape)
            # self.embeddings = torch.load(model_path)
            # print("embedding shape: ", self.embeddings.shape)
        elif name == "QM7b":
            graphs = to_networkx(QM7b(root=root_dir))
        elif name == "MUTAG":
            graphs = to_networkx(TUDataset(name='MUTAG', root=root_dir))
        elif name == "QM9":
            graphs = to_networkx(QM9(root=root_dir))
        elif name == "ZINC":
            graphs = to_networkx(ZINC(root=root_dir))
        
        if self.direct_embedding is False:        
            self.adj = []
            self.max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
            for g in graphs:
                adj_ = nx.to_numpy_matrix(g)
                self.adj.append(np.asarray(adj_) + np.identity(g.number_of_nodes()))
            if features == "id":
                self.features.append(np.identity(max_num_nodes))

            self.ae = torch.load(model_path).to(self.device)
            # freeze
            for param in self.ae.parameters():
                param.requires_grad = False
        
    def __len__(self):
        if self.direct_embedding:
            return len(self.embeddings)
        else:
            return len(self.adj)

    def __getitem__(self, idx):
        if self.direct_embedding:
            graph_embedding = self.embeddings[idx]
        else:
            adj = self.adj[idx]
            num_nodes = adj.shape[0]
            adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
            adj_padded[:num_nodes, :num_nodes] = adj
            features = self.features[idx]

            adj = torch.Tensor(adj_padded).to(self.device)
            feat = torch.Tensor(features).to(self.device)

            # {"adj": adj_padded, "adj_decoded": adj_vectorized, "features": features}
            # reshape
            graph_embedding = self.ae.encoder(feat, adj)
        graph_embedding = graph_embedding.view(-1, self.image_size, self.image_size)

        return graph_embedding
    
# ----------------------------------------------------------------------------------

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
        
    # Call our dataset ---------------------------------------------------------------------------------------
    if config.data.dataset in ["QM7b", "MUTAG", "QM9", "ZINC"]:
        # name, root_dir, model_path, image_size=
        dataset = GraphEmbeddings(name=config.data.dataset, root_dir=config.data.root_dir, model_path=config.data.model_path, embedding_path=config.data.embedding_path, image_size=config.data.image_size, device=config.device, direct_embedding_flag=config.data.direct_embedding_flag)
        # test_dataset = GraphEmbeddings(name=config.data.dataset, root_dir=config.data.root_dir, model_path=config.data.model_path, image_size=config.data.image_size, train=False, device=config.device)
        test_dataset = None
    # ---------------------------------------------------------------------------------------------------------
    elif config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(args.exp, "datasets", "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    elif config.data.dataset == "LSUN":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
            )

        test_dataset = LSUN(
            root=os.path.join(args.exp, "datasets", "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
        )

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
                ),
                resolution=config.data.image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.ToTensor(),
                resolution=config.data.image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
