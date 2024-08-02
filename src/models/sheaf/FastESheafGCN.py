import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric.utils import dropout_edge

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

    def forward(self, node_features, edge_index):
        # Calc sheaf matrix
        smat = torch.reshape(self.fc_smat(node_features), (-1, self.dimy, self.dimx))

        # Apply sheaf matrix
        q = torch.bmm(smat, node_features.unsqueeze(-1)).squeeze(-1)

        # Make message-passing step with LightGCN
        conv_emb = self.conv1(q, edge_index)

        return conv_emb


class FastESheafGCN(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset):
        super(FastESheafGCN, self).__init__()
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(dataset.num_users + dataset.num_items, latent_dim)
        self.num_nodes = dataset.num_items + dataset.num_users
        self.sheaf_conv1 = FastESheafLayer(latent_dim, latent_dim * 2, 40)

        self.train_edge_index = self.dataset.train_edge_index
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        emb0 = self.embedding.weight
        emb1 = self.sheaf_conv1(emb0, edge_index)

        return emb0, emb1


    def training_step(self, batch):
        users, pos_items, neg_items = batch
        edge_index, edge_mask = dropout_edge(self.train_edge_index)
        emb0, embs, users_emb, pos_emb, neg_emb = self.encode_minibatch(users,
                                                                        pos_items,
                                                                        neg_items,
                                                                        edge_index)
        bpr_loss = compute_bpr_loss(users, users_emb, pos_emb, neg_emb)
        loss = bpr_loss
        self.log('bpr_loss', bpr_loss)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, out = self.forward(edge_index)
        return (
            emb0,
            out,
            out[users],
            out[pos_items],
            out[neg_items],
        )
