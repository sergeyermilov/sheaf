from typing import Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from src.losses.bpr import compute_bpr_loss
from src.losses.sheaf import compute_loss_weights_simple
from src.losses import Losses


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
                 losses=None):
        super(ESheafGCN, self).__init__()
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.losses = losses

        if losses is None:
            self.losses = {Losses.BPR, Losses.DIFFUSION, Losses.ORTHOGONALITY, Losses.CONSISTENCY}
        else:
            self.losses = set(losses)

        Losses.validate(self.losses)

        if Losses.BPR not in self.losses:
            raise Exception("Missing BPR loss")

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

    def do_step(self, batch, batch_idx, suffix):
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

        self.log(f'{suffix}_bpr_loss', bpr_loss)
        self.log(f'{suffix}_loss_smap', loss_smap)
        self.log(f'{suffix}_loss_orth', loss_orth)
        self.log(f'{suffix}_loss_cons', loss_cons)

        self.log(f"{suffix}_w_smap", w_smap)
        self.log(f"{suffix}_w_bpr", w_bpr)
        self.log(f"{suffix}_w_bpr", w_orth)
        self.log(f"{suffix}_w_cons", w_cons)
        self.log(f'{suffix}_loss', loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

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
