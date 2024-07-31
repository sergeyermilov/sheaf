import torch
import pytorch_lightning as pl
from torch import nn

from src.losses.bpr import compute_bpr_loss, compute_loss_weights_simple


class Losses:
    ORTHOGONALITY = "orth"
    CONSISTENCY = "cons"


def debug_print_tensor(x, prefix):
    print("Tensor name: " + prefix)
    print(f"shape = {x.shape}")
    print(x)


class ESheafLayer(nn.Module):
    def __init__(self, dimx, dimy, nsmat=64):
        super(ESheafLayer, self).__init__()
        self.dimx = dimx
        self.dimy = dimy

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

    def forward(self, w, x, compute_loss=False):
        smat = torch.reshape(self.fc_smat(x), (-1, self.dimy, self.dimx))
        q = torch.bmm(smat, x.unsqueeze(-1)).squeeze(-1)

        if not compute_loss:
            return q

        y = torch.tensordot(w, q, dims=([1], [0]))
        xmap = torch.einsum('ijk,ij->ik', smat, y)
        rmat = torch.einsum('ijk,ilk->lji', smat, smat)

        target_matrix = torch.zeros((self.dimy, self.dimy, rmat.shape[-1]))

        for idx in range(target_matrix.shape[-1]):
             target_matrix[:, :, idx] = torch.eye(self.dimy)

        rmat = rmat - target_matrix
        smat_proj = torch.reshape(self.fc_smat(xmap), (-1, self.dimy, self.dimx))
        return xmap, q, rmat, smat, smat_proj


class ESheafGCN(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset,
                 losses):
        super(ESheafGCN, self).__init__()
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.losses = losses

        assert all([loss in {Losses.ORTHOGONALITY, Losses.CONSISTENCY} for loss in self.losses]), "unknown loss type"

        self.embedding = nn.Embedding(dataset.num_users + dataset.num_items, latent_dim)
        self.num_nodes = dataset.num_items + dataset.num_users
        self.sheaf_conv1 = ESheafLayer(latent_dim, latent_dim*2, 40)
        self.sheaf_conv2 = ESheafLayer(latent_dim*2, latent_dim*2, 40)
        self.sheaf_conv3 = ESheafLayer(latent_dim*2, latent_dim*2, 40)

        self.adj = self.dataset.adjacency_matrix
        degree = self.adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        diag_degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        self.normalized_adj = diag_degree_inv_sqrt @ self.adj @ diag_degree_inv_sqrt
        self.adj = self.normalized_adj
        self.train_edge_index = self.dataset.train_edge_index
        self.init_parameters()

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform(layer.weight)

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.sheaf_conv1.fc_smat.apply(self.init_weights)
        self.sheaf_conv2.fc_smat.apply(self.init_weights)
        self.sheaf_conv3.fc_smat.apply(self.init_weights)

    def forward_(self, adj_matrix):
        emb0 = self.embedding.weight
        xmap, y1, rmat, smat, smat_proj = self.sheaf_conv1(adj_matrix, emb0, True)
        y2 = self.sheaf_conv2(adj_matrix, y1, False)
        y3 = self.sheaf_conv3(adj_matrix, y2, False)

        return emb0, xmap, y1, y3, rmat, smat, smat_proj

    def forward(self, adj_matrix):
        emb0 = self.embedding.weight
        y1 = self.sheaf_conv1(adj_matrix, emb0, False)
        y2 = self.sheaf_conv2(adj_matrix, y1, False)
        y3 = self.sheaf_conv3(adj_matrix, y2, False)
        return emb0, y3

    def training_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        emb0, xmap, embs, users_emb, pos_emb, neg_emb, rmat, smat, smat_proj = self.encode_minibatch(users, pos_items, neg_items, self.adj)
        bpr_loss = compute_bpr_loss(users, users_emb, pos_emb, neg_emb)

        loss_smap = torch.mean((xmap - emb0) * (xmap - emb0)) * self.sheaf_conv1.dimx
        loss_orth = torch.sqrt(torch.mean(rmat * rmat) * self.latent_dim * self.latent_dim)
        loss_cons = torch.mean((smat_proj - smat) * (smat_proj - smat)) * self.latent_dim * self.latent_dim
        w_smap, w_orth, w_cons, w_bpr = compute_loss_weights_simple(loss_smap, loss_orth, loss_cons, bpr_loss, 1024)

        loss = w_smap * loss_smap + w_bpr * bpr_loss #+ w_orth * loss_orth + w_cons * loss_cons

        if Losses.CONSISTENCY in self.losses:
            loss += w_cons * loss_cons

        if Losses.ORTHOGONALITY in self.losses:
            loss += w_orth * loss_orth

        self.log('bpr_loss', bpr_loss)
        self.log('loss_smap', loss_smap)
        self.log('loss_orth', loss_orth)
        self.log('loss_cons', loss_cons)

        self.log("w_smap", w_smap)
        self.log("w_bpr", w_bpr)
        self.log("w_bpr", w_orth)
        self.log("w_cons", w_cons)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, xmap, y, out, rmat, smat, smat_proj = self.forward_(edge_index)
     #   debug_print_tensor(y, "Y")
     #   debug_print_tensor(pos_items, "pos_items")
     #   debug_print_tensor(neg_items, "neg_items")
      #  print(max(pos_items))
      #  print(max(neg_items))
        return (
           emb0,
           xmap,
           out,
           out[users],
           out[pos_items],
           out[neg_items],
           rmat,
           smat,
           smat_proj
        )
