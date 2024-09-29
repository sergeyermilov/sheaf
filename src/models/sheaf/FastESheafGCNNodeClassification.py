import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn.functional import one_hot
from torch_geometric.nn import SimpleConv

from src.losses.bpr import compute_bpr_loss
from src.models.graph.LightGCN import LightGCNConv


class FastESheafLayer(nn.Module):
    def __init__(self, dimx, dimy, nsmat=64):
        super(FastESheafLayer, self).__init__()
        self.dimx = dimx
        self.dimy = dimy

        self.conv1 = LightGCNConv()

        self.fc_smat = nn.Sequential(nn.Linear(self.dimx, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, self.dimy * self.dimx))

    def forward(self, node_features, compute_loss=False):
        # Calc sheaf matrix
        smat = torch.reshape(self.fc_smat(node_features), (-1, self.dimy, self.dimx))

        # Apply sheaf matrix
        q = torch.bmm(smat, node_features.unsqueeze(-1)).squeeze(-1)

        if not compute_loss:
            return q

        y = torch.tensordot(node_features, q, dims=([1], [0]))
        xmap = torch.einsum('ijk,ij->ik', smat, y)
        rmat = torch.einsum('ijk,ilk->lji', smat, smat)

        target_matrix = torch.zeros((self.dimy, self.dimy, rmat.shape[-1]))

        for idx in range(target_matrix.shape[-1]):
             target_matrix[:, :, idx] = torch.eye(self.dimy)

        rmat = rmat - target_matrix
        smat_proj = torch.reshape(self.fc_smat(xmap), (-1, self.dimy, self.dimx))
        return xmap, q, rmat, smat, smat_proj


class FastESheafGCNNodeClassification(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset):
        super(FastESheafGCNNodeClassification, self).__init__()
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.node_features = self.dataset.node_features
        self.node_labels = self.dataset.node_labels

        self.sheaf = FastESheafLayer(self.dataset.num_features, latent_dim, 40)
        self.conv1 = LightGCNConv()
        self.conv2 = LightGCNConv()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, self.dataset.num_classes)

        self.train_edge_index = self.dataset.train_edge_index
        self.loss = nn.CrossEntropyLoss()
        self.init_parameters()

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=0.1)

    def init_parameters(self):
        self.sheaf.fc_smat.apply(self.init_weights)
        nn.init.normal_(self.fc1.weight, std=0.1)
        nn.init.normal_(self.fc2.weight, std=0.1)


    def forward(self):
        emb1 = self.sheaf(self.node_features)
        emb2 = self.conv1(emb1, self.train_edge_index)
        out = self.conv2(emb2, self.train_edge_index)
        logits = self.fc2(self.relu(self.fc1(out)))
        return logits


    def training_step(self, batch):
        indexes, features = batch
        proba = self.encode_minibatch()

        cross_entropy_loss = self.loss(proba[indexes],
                                       one_hot(self.node_labels[indexes],
                                               self.dataset.num_classes)
                                       .double()
                                       )
        return cross_entropy_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def encode_minibatch(self):
        logits = self.forward()
        return logits
