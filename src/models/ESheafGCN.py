import torch
import pytorch_lightning as pl
from torch import nn

from src.losses.bpr import compute_bpr_loss, compute_loss_weights_simple


def debug_print_tensor(x, prefix):
    print("Tensor name: " + prefix)
    print(f"shape = {x.shape}")
    print(x)

class Sheaf_Conv_fixed(nn.Module):
    def __init__(self, dimx, dimy, nsmat=40):
        super(Sheaf_Conv_fixed, self).__init__()
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

    def forward(self, w, x):
        smat = torch.reshape(self.fc_smat(x), (-1, self.dimy, self.dimx))
        q = torch.bmm(smat, x.unsqueeze(-1)).squeeze(-1)
        y = torch.tensordot(w, q, dims=([1], [0]))
        xmap = torch.einsum('ijk,ij->ik', smat, y)
        return xmap, q


class ESheafGCN(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset):
        super(ESheafGCN, self).__init__()
        self.dataset = dataset
        self.embedding = nn.Embedding(dataset.num_users + dataset.num_items, latent_dim)
        self.num_nodes = dataset.num_items + dataset.num_users
        self.sheaf_conv1 = Sheaf_Conv_fixed(latent_dim, latent_dim * 2, 40)
        self.sheaf_conv2 = Sheaf_Conv_fixed(latent_dim, latent_dim * 2, 40)
        self.sheaf_conv3 = Sheaf_Conv_fixed(latent_dim, latent_dim * 2, 40)
        self.adj = self.dataset.adjacency_matrix
        degree = self.adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        diag_degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        self.normalized_adj = diag_degree_inv_sqrt @ self.adj @ diag_degree_inv_sqrt
        self.adj = self.normalized_adj
        self.init_parameters()

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform(layer.weight)

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.sheaf_conv1.fc_smat.apply(self.init_weights)
        self.sheaf_conv2.fc_smat.apply(self.init_weights)
        self.sheaf_conv3.fc_smat.apply(self.init_weights)

    def forward(self, adj_matrix):
       emb0 = self.embedding.weight
       xmap, y = self.sheaf_conv1(adj_matrix, emb0)
       return emb0, xmap, y

    def training_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        emb0, xmap, embs, users_emb, pos_emb, neg_emb = self.encode_minibatch(users, pos_items, neg_items, self.adj)
        bpr_loss = compute_bpr_loss(users, users_emb, pos_emb, neg_emb)

        loss_smap = torch.mean((xmap - emb0) * (xmap - emb0)) * self.sheaf_conv1.dimx

        w_smap, w_bpr = compute_loss_weights_simple(loss_smap, bpr_loss, 1024)
        loss = w_smap * loss_smap + w_bpr * bpr_loss
        self.log('loss_smap', loss_smap)
        self.log('bpr_loss', bpr_loss)
        self.log('loss', loss)
        self.log("w_smap", w_smap)
        self.log("w_bpr", w_bpr)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, xmap, y = self.forward(edge_index)
        return (
           emb0,
           xmap,
           y,
           y[users],
           y[pos_items],
           y[neg_items]
        )
    



class ESheafGCN_content_features(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset):
        super(ESheafGCN_content_features, self).__init__()
        self.dataset = dataset

        self.user_embedding = [nn.Embedding(num_unique, latent_dim) for num_unique in self.dataset.embed_layers_config['users']]
        self.item_embedding = [nn.Embedding(num_unique, latent_dim) for num_unique in self.dataset.embed_layers_config['items']]

        self.num_nodes = dataset.num_items + dataset.num_users
        self.sheaf_conv1 = Sheaf_Conv_fixed(latent_dim, latent_dim * 2, 40)
        self.sheaf_conv2 = Sheaf_Conv_fixed(latent_dim, latent_dim * 2, 40)
        self.sheaf_conv3 = Sheaf_Conv_fixed(latent_dim, latent_dim * 2, 40)
        self.adj = self.dataset.adjacency_matrix
        degree = self.adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        diag_degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        self.normalized_adj = diag_degree_inv_sqrt @ self.adj @ diag_degree_inv_sqrt
        self.adj = self.normalized_adj
        self.init_parameters()

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform(layer.weight)

    def init_parameters(self):
        for i in range(len(self.user_embedding)):
            nn.init.normal_(self.user_embedding[i].weight, std=0.1)

        for i in range(len(self.item_embedding)):
            nn.init.normal_(self.item_embedding[i].weight, std=0.1)

        self.sheaf_conv1.fc_smat.apply(self.init_weights)
        self.sheaf_conv2.fc_smat.apply(self.init_weights)
        self.sheaf_conv3.fc_smat.apply(self.init_weights)

    def forward(self, adj_matrix, emb0):
 
        xmap, y = self.sheaf_conv1(adj_matrix, emb0)
        return xmap, y

    def training_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        emb0, xmap, embs, users_emb, pos_emb, neg_emb = self.encode_minibatch(users, pos_items, neg_items, self.adj)
        bpr_loss = compute_bpr_loss(users, users_emb, pos_emb, neg_emb)

        # loss_smap = torch.mean((xmap - emb0) * (xmap - emb0)) * self.sheaf_conv1.dimx

        # w_smap, w_bpr = compute_loss_weights_simple(loss_smap, bpr_loss, 1024)
        loss = bpr_loss #w_smap * loss_smap + w_bpr * bpr_loss
        self.log('bpr_loss', bpr_loss)

        # self.log('loss_smap', loss_smap)
        # self.log('bpr_loss', bpr_loss)
        # self.log('loss', loss)
        # self.log("w_smap", w_smap)
        # self.log("w_bpr", w_bpr)
        return loss

    def configure_optimizers(self):
        all_params = []

        for embedder in self.user_embedding:
            for p in embedder.parameters():
                all_params.append(p)

        for embedder in self.item_embedding:
            for p in embedder.parameters():
                all_params.append(p)

        all_params += list(self.parameters())

        print(all_params)

        return torch.optim.Adam(all_params, lr=0.0001)

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0 = self.eval_emb()

        xmap, y = self.forward(edge_index, emb0)
        return (
           emb0,
           xmap,
           y,
           y[users],
           y[pos_items],
           y[neg_items]
        )
    
    def eval_emb(self):
        users_raw = self.dataset.raw_embeds['users']
        users_emb = torch.stack([self.user_embedding[i](cur_feat_idxs) for i, cur_feat_idxs in enumerate(users_raw.T.split(1))]).mean(dim=0)

        items_raw = self.dataset.raw_embeds['items']
        genres_feat = items_raw[:, 1:]
        genres_embeds = torch.stack([self.item_embedding[1](torch.where(row == 1)[0]).mean(dim=0) for row in genres_feat])
        items_emb = torch.stack([self.item_embedding[0](items_raw[:, 1]), genres_embeds]).mean(dim=0)

        return torch.vstack([users_emb[0], items_emb])