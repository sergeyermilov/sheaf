import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import GATConv

from src.losses.bpr import compute_bpr_loss_with_reg


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
        self.adj = self.dataset.adjacency_matrix
        degree = self.adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        diag_degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        self.normalized_adj = diag_degree_inv_sqrt @ self.adj @ diag_degree_inv_sqrt
        self.adj = self.normalized_adj
        self.train_edge_index = self.dataset.train_edge_index
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)


    def forward(self, adj_matrix):
       emb0 = self.embedding.weight
       emb1 = self.conv1(emb0, self.train_edge_index)
       emb2 = self.conv2(emb1, self.train_edge_index)
       emb3 = self.conv3(emb2, self.train_edge_index)
       emb4 = self.conv4(emb3, self.train_edge_index)
       emb5 = self.conv5(emb4, self.train_edge_index)
       embs = [emb0, emb1, emb2, emb3, emb4, emb5]
       out = (torch.mean(torch.stack(embs, dim=0), dim=0))
       return emb0, out

    def training_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = self.encode_minibatch(users, pos_items, neg_items, self.adj)
        bpr_loss, reg_loss = compute_bpr_loss_with_reg(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)

        loss = bpr_loss + reg_loss
        self.log('bpr_loss', bpr_loss)
        self.log('loss', loss)
        return loss

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
