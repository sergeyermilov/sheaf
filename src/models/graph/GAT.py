import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj

from src.losses.bpr import compute_bpr_loss_with_reg
from src.models.graph_utils import compute_adj_normalized


def debug_print_tensor(x, prefix):
    print("Tensor name: " + prefix)
    print(f"shape = {x.shape}")
    print(x)


class GAT(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset):
        super(GAT, self).__init__()
        self.dataset = dataset

        self.embedding = nn.Embedding(dataset.num_users + dataset.num_items, latent_dim)
        self.num_nodes = dataset.num_items + dataset.num_users

        self.conv1 = GATConv(latent_dim, latent_dim, 1, )
        self.conv2 = GATConv(latent_dim, latent_dim, 1, )
        self.conv3 = GATConv(latent_dim, latent_dim, 1, )
        self.conv4 = GATConv(latent_dim, latent_dim, 1, )
        self.conv5 = GATConv(latent_dim, latent_dim, 1, )

        self.edge_index = self.dataset.train_edge_index
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index: torch.Tensor):
        emb0 = self.embedding.weight

        emb1 = self.conv1(emb0, edge_index)
        emb2 = self.conv2(emb1, edge_index)
        emb3 = self.conv3(emb2, edge_index)
        emb4 = self.conv4(emb3, edge_index)
        emb5 = self.conv5(emb4, edge_index)
        embs = [emb0, emb1, emb2, emb3, emb4, emb5]
        out = (torch.mean(torch.stack(embs, dim=0), dim=0))
        return emb0, out

    def do_step(self, batch, batch_idx, suffix):
        if len(batch) == 3:
            start_nodes, pos_items, neg_items = batch
            edge_index = self.edge_index
        elif len(batch) == 4:
            start_nodes, pos_items, neg_items, edge_index = batch
        else:
            raise Exception("batch is of unknown size")

        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = self.encode_minibatch(start_nodes, pos_items, neg_items, edge_index)
        bpr_loss, reg_loss = compute_bpr_loss_with_reg(start_nodes, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)

        loss = bpr_loss + reg_loss
        self.log(f'{suffix}_bpr_loss', bpr_loss)
        self.log(f'{suffix}_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, out = self.forward(edge_index)
     #   debug_print_tensor(y, "Y")
     #   debug_print_tensor(pos_items, "pos_items")
     #   debug_print_tensor(neg_items, "neg_items")
      #  print(max(pos_items))
      #  print(max(neg_items))
        return (
           out[users],
           out[pos_items],
           out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items]
        )
