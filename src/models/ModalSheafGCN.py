import torch
import pytorch_lightning as pl
from torch import nn

from src.losses.bpr import compute_bpr_loss, compute_loss_weights_simple, compute_bpr_loss_with_reg

"This approach works only for bipartite graphs with two entities!"


class Sheaf_Conv_fixed(nn.Module):
    def __init__(self, dimx, dimy):
        super(Sheaf_Conv_fixed, self).__init__()
        self.dimx = dimx
        self.dimy = dimy

        self.user_operator = nn.Linear(self.dimx, self.dimy)
        self.item_operator = nn.Linear(self.dimx, self.dimy)

    def forward(self, adj_matrix, embeddings, edge_index, compute_losses: bool = False):
        # first half is users
        # second half is objects
        entity_separation = embeddings.shape[0] // 2

        u = edge_index[0, :]
        v = edge_index[1, :]

        e_u = embeddings[u, :]
        e_v = embeddings[v, :]

        e_v_item = e_v[:entity_separation, :]
        e_v_user = e_v[entity_separation:, :]

        h_v_user = self.user_operator(e_v_user)
        h_v_item = self.item_operator(e_v_item)

        h_user = torch.matmul(self.item_operator.weight.T, h_v_user.T)
        h_item = torch.matmul(self.user_operator.weight.T, h_v_item.T)

        e_embedds = torch.concat([h_user.T, h_item.T], axis=0)

        # compute c_v = w*(v,u) * h_v
        embedding_weights = adj_matrix[edge_index[0, :], edge_index[1, :]]
        c_v = embedding_weights.view(-1, 1) * e_embedds

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

        # # compute intermediate values for loss cons
        p_item = torch.matmul(self.item_operator.weight.T, self.item_operator.weight)
        p_user = torch.matmul(self.user_operator.weight.T, self.user_operator.weight)

        cons_y_item = self.item_operator.weight - torch.matmul(self.item_operator.weight, p_item)
        cons_y_user = self.user_operator.weight - torch.matmul(self.user_operator.weight, p_user)

        cons_q_item = torch.matmul(cons_y_item.T, cons_y_item)
        cons_q_user = torch.matmul(cons_y_user.T, cons_y_user)

        cons_user_idx = torch.unique(u[:entity_separation])
        cons_item_idx = torch.unique(u[entity_separation:])

        cons_emb_user = embeddings[cons_user_idx, :]
        cons_emb_item = embeddings[cons_item_idx, :]

        cons_w1_user = torch.matmul(cons_emb_user, cons_q_user)
        cons_w2_user = (cons_w1_user + cons_w1_user).sum(axis=-1)

        cons_w1_item = torch.matmul(cons_emb_item, cons_q_item)
        cons_w2_item = (cons_w1_item + cons_w1_item).sum(axis=-1)

        cons_loss = torch.mean(torch.concat([cons_w2_user, cons_w2_item]))

        # compute intermediate values for loss orth
        a = torch.matmul(self.item_operator.weight, self.item_operator.weight.T) - torch.eye(
            self.item_operator.weight.shape[0])
        b = torch.matmul(self.user_operator.weight, self.user_operator.weight.T) - torch.eye(
            self.user_operator.weight.shape[0])

        orth_loss = (torch.trace(torch.matmul(a.T, a)) + torch.trace(torch.matmul(b.T, b))) / 2

        return m_u, diff_loss, cons_loss, orth_loss


class ModalSheafGCN(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset):
        super(ModalSheafGCN, self).__init__()
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(dataset.num_users + dataset.num_items, latent_dim)
        self.num_nodes = dataset.num_items + dataset.num_users

        self.sheaf_conv3 = Sheaf_Conv_fixed(latent_dim, latent_dim)
        self.sheaf_conv2 = Sheaf_Conv_fixed(latent_dim, latent_dim)
        self.sheaf_conv1 = Sheaf_Conv_fixed(latent_dim, latent_dim)

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
            nn.init.xavier_uniform(layer.weight)

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)

        self.sheaf_conv1.user_operator.apply(self.init_weights)
        self.sheaf_conv1.item_operator.apply(self.init_weights)

        self.sheaf_conv2.user_operator.apply(self.init_weights)
        self.sheaf_conv2.item_operator.apply(self.init_weights)

        self.sheaf_conv3.user_operator.apply(self.init_weights)
        self.sheaf_conv3.item_operator.apply(self.init_weights)

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
