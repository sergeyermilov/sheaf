import torch
import pytorch_lightning as pl
from torch import nn

from src.losses.bpr import compute_bpr_loss, compute_loss_weights_simple

"""
This is extension over an approach implemented in EXSheafGCN. Here we use FFN over two embeddings and two global matrices
to compute linear operator. A = FFN(u, v) + Q_user if u is user node, and A = FFN(u, v) + Q_item if u is item node.  
"""

class Sheaf_Conv_fixed(nn.Module):
    def __init__(self, dimx, dimy, nsmat=64):
        super(Sheaf_Conv_fixed, self).__init__()
        self.dimx = dimx
        self.dimy = dimy

        self.orth_eye = torch.eye(self.dimy).unsqueeze(0)

        self.user_operator = nn.Parameter(torch.zeros((self.dimy, self.dimx)), requires_grad=True)
        self.item_operator = nn.Parameter(torch.zeros((self.dimy, self.dimx)), requires_grad=True)
        self.fc_smat = nn.Sequential(nn.Linear(self.dimx * 2, nsmat),
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

    def forward(self, adj_matrix, embeddings, edge_index, compute_losses: bool = False):
        # first half is users
        # second half is objects
        entity_separation = edge_index.shape[0] // 2

        u = edge_index[0, :]
        v = edge_index[1, :]

        e_u = embeddings[u, :]
        e_v = embeddings[v, :]

        comb_uv = torch.concat([e_u, e_v], axis=-1)  # u, v -> A(u, v)
        comb_vu = torch.concat([e_v, e_u], axis=-1)  # v, u -> A(v, u)

        smat_uv = torch.reshape(self.fc_smat(comb_uv), (-1, self.dimy, self.dimx))  # A(u, v)
        smat_uv[:entity_separation, ...] += self.user_operator
        smat_uv[entity_separation:, ...] += self.item_operator
        smat_uv_t = torch.reshape(smat_uv, (-1, self.dimy, self.dimx)).swapaxes(-1, -2)  # A(u, v)^T

        smat_vu = torch.reshape(self.fc_smat(comb_vu), (-1, self.dimy, self.dimx))  # A(v, u)
        smat_vu[entity_separation:, ...] += self.user_operator
        smat_vu[:entity_separation, ...] += self.item_operator

        # compute h_v = A(u,v)^T A(v,u) * x(v)
        h_v_ = torch.bmm(smat_vu, e_v.unsqueeze(-1)).squeeze(-1)
        h_v = torch.bmm(smat_uv_t, h_v_.unsqueeze(-1)).squeeze(-1)

        # compute c_v = w*(v,u) * h_v
        embedding_weights = adj_matrix[edge_index[0, :], edge_index[1, :]]
        c_v = embedding_weights.view(-1, 1) * h_v

        # compute  sum_v
        m_u = torch.zeros_like(embeddings)
        indx = u.view(-1, 1).repeat(1, embeddings.shape[1])
        # sum c_v for each u
        m_u = torch.scatter_reduce(input=m_u, src=c_v, index=indx, dim=0, reduce="sum", include_self=False)

        if not compute_losses:
            return m_u, None, None, None

        # compute intermediate values for loss diff
        diff_x = (m_u - embeddings).unsqueeze(-1)
        diff_w = torch.bmm(diff_x.swapaxes(-1, -2), diff_x)
        diff_loss = diff_w.mean()

        # compute intermediate values for loss cons
        # P(u, v) = A(u, v)^T A(u, v)
        cons_p = torch.bmm(smat_uv_t, smat_uv)
        # A(u, v) - A(u, v) P(u, v)
        cons_y = smat_uv - torch.bmm(smat_uv, cons_p)
        # Q(u, v) = (A(u, v) - A(u, v) P(u, v))^T (A(u, v) - A(u, v) P(u, v))
        cons_q = torch.bmm(cons_y.swapaxes(-1, -2), cons_y)
        # W(u, v) = x(u)^T Q(u, v) x(u)
        cons_w1 = torch.bmm(cons_q, e_u.unsqueeze(-1))
        cons_w2 = torch.bmm(e_u.unsqueeze(-1).swapaxes(-1, -2), cons_w1)
        cons_loss = cons_w2.mean()

        # compute intermediate values for loss orth
        orth_aat = torch.bmm(smat_uv, smat_uv_t)
        orth_q = orth_aat - self.orth_eye
        orth_z = torch.bmm(orth_q.swapaxes(-1, -2), orth_q)

        # compute trace
        orth = torch.einsum("ijj", orth_z)
        orth_loss = torch.mean(orth)

        return m_u, diff_loss, cons_loss, orth_loss


class BimodalEXSheafGCN(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset):
        super(BimodalEXSheafGCN, self).__init__()
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(dataset.num_users + dataset.num_items, latent_dim)
        self.num_nodes = dataset.num_items + dataset.num_users

        self.sheaf_conv1 = Sheaf_Conv_fixed(latent_dim, latent_dim, 40)
        self.sheaf_conv2 = Sheaf_Conv_fixed(latent_dim, latent_dim, 40)
        self.sheaf_conv3 = Sheaf_Conv_fixed(latent_dim, latent_dim, 40)

        self.edge_index = self.dataset.train_edge_index
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
            nn.init.xavier_uniform_(layer.weight)

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.sheaf_conv1.fc_smat.apply(self.init_weights)
        self.sheaf_conv2.fc_smat.apply(self.init_weights)
        self.sheaf_conv3.fc_smat.apply(self.init_weights)
        nn.init.xavier_uniform_(self.sheaf_conv1.user_operator.data)
        nn.init.xavier_uniform_(self.sheaf_conv1.item_operator.data)
        nn.init.xavier_uniform_(self.sheaf_conv2.user_operator.data)
        nn.init.xavier_uniform_(self.sheaf_conv2.item_operator.data)
        nn.init.xavier_uniform_(self.sheaf_conv3.user_operator.data)
        nn.init.xavier_uniform_(self.sheaf_conv3.item_operator.data)

    def forward(self, adj_matrix):
        emb0 = self.embedding.weight
        m_u0, diff_loss, cons_loss, orth_loss = self.sheaf_conv1(adj_matrix, emb0, self.edge_index, True)
        m_u1, _, _, _ = self.sheaf_conv2(adj_matrix, m_u0, self.edge_index)
        out, _, _, _ = self.sheaf_conv3(adj_matrix, m_u1, self.edge_index)

        return out, diff_loss, cons_loss, orth_loss

    def training_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        embs, users_emb, pos_emb, neg_emb, loss_diff, loss_cons, loss_orth = self.encode_minibatch(users, pos_items,
                                                                                                   neg_items, self.adj)
        bpr_loss = compute_bpr_loss(users, users_emb, pos_emb, neg_emb)

        w_diff, w_orth, w_cons, w_bpr = compute_loss_weights_simple(loss_diff, loss_orth, loss_cons, bpr_loss, 1024)

        loss = w_diff * loss_diff + w_orth * loss_orth + w_cons * loss_cons + w_bpr * bpr_loss

        self.log('bpr_loss', bpr_loss)
        self.log('loss_diff', loss_diff)
        self.log('loss_orth', loss_orth)
        self.log('loss_cons', loss_cons)

        self.log("w_diff", w_diff)
        self.log("w_bpr", w_bpr)
        self.log("w_bpr", w_orth)
        self.log("w_cons", w_cons)
        self.log('loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        out, diff_loss, cons_loss, orth_loss = self.forward(edge_index)

        return (
            out,
            out[users],
            out[pos_items],
            out[neg_items],
            diff_loss,
            cons_loss,
            orth_loss
        )
