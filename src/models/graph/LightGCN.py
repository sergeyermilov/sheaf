
import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

from src.losses.bpr import compute_bpr_loss_with_reg


class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class LightGCN(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset):
        super(LightGCN, self).__init__()
        self.dataset = dataset
        self.embedding = nn.Embedding(dataset.num_users + dataset.num_items, latent_dim)
        self.conv1 = LightGCNConv()
        self.conv2 = LightGCNConv()
        self.conv3 = LightGCNConv()
        self.edge_index = self.dataset.train_edge_index
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)

    # compatibility with common graph nnet interface
    def forward(self, edge_index: torch.Tensor):
       emb0 = self.embedding.weight
       emb1 = self.conv1(emb0, edge_index)
       emb2 = self.conv2(emb1, edge_index)
       emb3 = self.conv3(emb2, edge_index)

       embs = [emb0, emb1, emb2, emb3]
       out = (torch.mean(torch.stack(embs, dim=0), dim=0))
       return emb0, out

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            start_nodes, pos_items, neg_items = batch
            edge_index = self.edge_index
        elif len(batch) == 4:
            start_nodes, pos_items, neg_items, edge_index = batch
        else:
            raise Exception("batch is of unknown size")

        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = self.encode_minibatch(start_nodes, pos_items, neg_items, edge_index)
        bpr_loss, reg_loss = compute_bpr_loss_with_reg(start_nodes, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)
        final_loss = bpr_loss + reg_loss
        self.log('final_loss', final_loss)
        self.log('bpr_loss', bpr_loss)
        self.log('reg_loss', reg_loss)
        return final_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, out = self.forward(edge_index)
        return (
           out[users],
           out[pos_items],
           out[neg_items],
           emb0[users],
           emb0[pos_items],
           emb0[neg_items])
