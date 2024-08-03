import dataclasses

import torch
import pytorch_lightning as pl
from torch import nn

from src.losses.bpr import compute_bpr_loss, compute_loss_weights_simple

"""
This is extension over an approach implemented in EXSheafGCN. Here we use FFN over two embeddings and two global matrices
to compute linear operator. A(u, v) = FFN(u, v) + FFN(u) + U for u and v vectors (and vise versa).  
"""


class Losses:
    ORTHOGONALITY = "orth"
    CONSISTENCY = "cons"


def make_fc_transform(inpt: int, outpt: tuple[int, int], nsmat: int):
    assert len(outpt) == 2, "incorrect output dim"

    return nn.Sequential(
        nn.Linear(inpt, nsmat),
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
        nn.Linear(nsmat, outpt[0] * outpt[1])
    )


class OperatorComputeLayerType:
    LAYER_GLOBAL = "global"
    LAYER_SINGLE_ENTITY = "single"
    LAYER_PAIRED_ENTITIES = "paired"


@dataclasses.dataclass
class SheafOperators:
    operator_uv: torch.Tensor  # A(u, v)
    operator_vu: torch.Tensor  # A(v, u)


class OperatorComputeLayer(nn.Module):
    def __init__(self, dimx: int, dimy: int, user_indices: list[int], item_indices: list[int]):
        super(OperatorComputeLayer, self).__init__()

        self.dimx = dimx
        self.dimy = dimy

        self.user_indices = torch.tensor(user_indices)
        self.item_indices = torch.tensor(item_indices)

    def assert_operators(self, operator: torch.Tensor):
        assert (operator.shape[-2], operator.shape[-1]) == (self.dimy, self.dimx), "invalid shape"

    def assert_indices(self, indices: torch.Tensor, embeddings: torch.Tensor):
        assert torch.max(indices) < embeddings.shape[0], "invalid indices"

    def forward(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:
        self.assert_operators(operators.operator_uv)
        self.assert_operators(operators.operator_vu)

        self.assert_indices(u_indices, embeddings)
        self.assert_indices(v_indices, embeddings)

        return self.compute(operators, embeddings, u_indices, v_indices)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:
        raise NotImplementedError()

    def init_parameters(self):
        raise NotImplementedError()

    @staticmethod
    def init_layer(layer):
        if layer is nn.Linear:
            nn.init.xavier_uniform(layer.weight)


class GlobalOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dimx: int, dimy: int, user_indices: list[int], item_indices: list[int]):
        super(GlobalOperatorComputeLayer, self).__init__(dimx, dimy, user_indices, item_indices)

        self.user_operator = nn.Parameter(torch.zeros((self.dimy, self.dimx)), requires_grad=True)
        self.item_operator = nn.Parameter(torch.zeros((self.dimy, self.dimx)), requires_grad=True)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:
        operators.operator_uv[torch.isin(u_indices, self.user_indices), ...] += self.user_operator
        operators.operator_uv[torch.isin(u_indices, self.item_indices), ...] += self.item_operator
        operators.operator_vu[torch.isin(v_indices, self.user_indices), ...] += self.user_operator
        operators.operator_vu[torch.isin(v_indices, self.item_indices), ...] += self.item_operator
        return operators

    def init_parameters(self):
        nn.init.xavier_uniform(self.user_operator.data)
        nn.init.xavier_uniform(self.item_operator.data)


class SingleEntityOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dimx: int, dimy: int, user_indices: list[int], item_indices: list[int], nsmat: int = 64):
        super(SingleEntityOperatorComputeLayer, self).__init__(dimx, dimy, user_indices, item_indices)

        # maybe create two selarate FFNs for user and item nodes?
        self.fc_smat = make_fc_transform(self.dimx, (self.dimx, self.dimy), nsmat)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:

        operator_by_embedding = torch.reshape(self.fc_smat(embeddings), (-1, self.dimy, self.dimx))

        operators.operator_uv += operator_by_embedding[u_indices, ...]
        operators.operator_vu += operator_by_embedding[v_indices, ...]

        return operators

    def init_parameters(self):
        self.fc_smat.apply(OperatorComputeLayer.init_layer)


class PairedEntityOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dimx: int, dimy: int, user_indices: list[int], item_indices: list[int], nsmat: int = 32):
        super(PairedEntityOperatorComputeLayer, self).__init__(dimx, dimy, user_indices, item_indices)

        self.dimx = dimx
        self.dimy = dimy

        # maybe create two selarate FFNs for user and item nodes?
        self.fc_smat = make_fc_transform(self.dimx * 2, (self.dimx, self.dimy), nsmat)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:

        u_embeddings = embeddings[u_indices, ...]
        v_embeddings = embeddings[v_indices, ...]

        combined_embeddings_uv = torch.concat([u_embeddings, v_embeddings], dim=-1)
        combined_embeddings_vu = torch.concat([v_embeddings, u_embeddings], dim=-1)

        operator_uv = torch.reshape(self.fc_smat(combined_embeddings_uv), (-1, self.dimy, self.dimx))
        operator_vu = torch.reshape(self.fc_smat(combined_embeddings_vu), (-1, self.dimy, self.dimx))

        operators.operator_uv += operator_uv
        operators.operator_vu += operator_vu

        return operators

    def init_parameters(self):
        self.fc_smat.apply(OperatorComputeLayer.init_layer)


class ExtendableSheafGCNLayer(nn.Module):
    def __init__(self, dimx: int, dimy: int, operator_compute_layers: list[OperatorComputeLayer]):
        super(ExtendableSheafGCNLayer, self).__init__()
        self.dimx = dimx
        self.dimy = dimy
        self.operator_compute_layers = nn.ModuleList(operator_compute_layers)

        self.orth_eye = torch.eye(self.dimy).unsqueeze(0)

    @staticmethod
    def compute_sheaf(A_uv_t, A_v_u, embeddings, indices) -> torch.Tensor:
        #########################################
        ## compute h_v = A(u,v)^T * A(v,u) * x(v)
        #########################################
        x_v = embeddings[indices, ...]
        # compute A(v,u) * x(v)
        h_v_ = torch.bmm(A_v_u, x_v.unsqueeze(-1))
        # compute h_v = A(u,v)^T * A(v,u) * x(v)
        h_v = torch.bmm(A_uv_t, h_v_).squeeze(-1)
        #########################################

        return h_v

    @staticmethod
    def scale_sheaf(adj_matrix, u_indices, v_indices, h_v) -> torch.Tensor:
        ############################
        # compute c_v = w(v,u) * h_v
        ############################
        # extract w(v, u)
        embedding_weights = adj_matrix[v_indices, u_indices]
        # c_v = w(v, u) * h_v
        c_v = embedding_weights.view(-1, 1) * h_v
        #########################################

        return c_v

    @staticmethod
    def compute_message(embeddings, u_indices, sheafs):
        ############################
        # compute  sum_v
        ############################
        m_u = torch.zeros_like(embeddings)
        indx = u_indices.view(-1, 1).repeat(1, embeddings.shape[1])
        # sum c_v for each u
        return torch.scatter_reduce(
            input=m_u,
            src=sheafs,
            index=indx,
            dim=0,
            reduce="sum",
            include_self=False
        )

    @staticmethod
    def compute_diff_loss(messages, embeddings):
        diff_x = (messages - embeddings)
        diff_x_t = diff_x.swapaxes(-1, -2)
        diff_w = torch.mm(diff_x_t, diff_x)

        return diff_w.mean()

    @staticmethod
    def compute_cons_loss(embeddings, u_indices, A_uv, A_uv_t):
        embeddings_u = embeddings[u_indices, ...]
        x = embeddings_u.unsqueeze(-1)
        x_t = embeddings_u.unsqueeze(-1).swapaxes(-1, -2)

        # P(u, v) = A(u, v)^T A(u, v)
        cons_p = torch.bmm(A_uv_t, A_uv)
        # A(u, v) - A(u, v) P(u, v)
        cons_y = A_uv - torch.bmm(A_uv, cons_p)
        # Q(u, v) = (A(u, v) - A(u, v) P(u, v))^T (A(u, v) - A(u, v) P(u, v))
        cons_q = torch.bmm(cons_y.swapaxes(-1, -2), cons_y)
        # W(u, v) = x(u)^T Q(u, v) x(u)
        cons_w1 = torch.bmm(cons_q, x)
        cons_w2 = torch.bmm(x_t, cons_w1)

        return cons_w2.mean()

    @staticmethod
    def compute_orth_loss(A_uv, A_uv_t, orth_eye):
        # compute intermediate values for loss orth
        orth_aat = torch.bmm(A_uv, A_uv_t)
        orth_q = orth_aat - orth_eye
        orth_z = torch.bmm(orth_q.swapaxes(-1, -2), orth_q)

        # compute trace
        orth = torch.einsum("ijj", orth_z)

        return torch.mean(orth)

    def forward(self, adj_matrix, embeddings, edge_index, compute_losses: bool = False):
        u_indices = edge_index[0, :]
        v_indices = edge_index[1, :]

        sheaf_operators = SheafOperators(
            torch.zeros((edge_index.shape[1], self.dimy, self.dimx), requires_grad=False),
            torch.zeros((edge_index.shape[1], self.dimy, self.dimx), requires_grad=False)
        )

        for operator_compute_layer in self.operator_compute_layers:
            sheaf_operators = operator_compute_layer(sheaf_operators, embeddings, u_indices, v_indices)

        A_uv = sheaf_operators.operator_uv
        A_vu = sheaf_operators.operator_vu

        A_uv_t = torch.reshape(A_uv, (-1, self.dimy, self.dimx)).swapaxes(-1, -2)  # A(u, v)^T

        h_v = ExtendableSheafGCNLayer.compute_sheaf(A_uv_t=A_uv_t, A_v_u=A_vu, embeddings=embeddings, indices=v_indices)
        c_v = ExtendableSheafGCNLayer.scale_sheaf(adj_matrix=adj_matrix, u_indices=u_indices, v_indices=v_indices, h_v=h_v)
        m_u = ExtendableSheafGCNLayer.compute_message(embeddings=embeddings, u_indices=u_indices, sheafs=c_v)

        if not compute_losses:
            return m_u

        diff_loss = ExtendableSheafGCNLayer.compute_diff_loss(messages=m_u, embeddings=embeddings)
        cons_loss = ExtendableSheafGCNLayer.compute_cons_loss(embeddings=embeddings, u_indices=u_indices, A_uv=A_uv, A_uv_t=A_uv_t)
        orth_loss = ExtendableSheafGCNLayer.compute_orth_loss(A_uv=A_uv, A_uv_t=A_uv_t, orth_eye=self.orth_eye)

        return m_u, diff_loss, cons_loss, orth_loss

    def init_parameters(self):
        for layer in self.operator_compute_layers:
            layer.init_parameters()


class ExtendableSheafGCN(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset,
                 layer_types: list[str] = None,
                 losses: list[str] = None):
        super(ExtendableSheafGCN, self).__init__()

        if layer_types is None:
            layer_types = [OperatorComputeLayerType.LAYER_SINGLE_ENTITY]

        if losses is None:
            self.losses = {}
        else:
            self.losses = set(losses)

        assert all([loss in {Losses.ORTHOGONALITY, Losses.CONSISTENCY} for loss in self.losses]), "unknown loss type"
        assert layer_types, "layers may not be empty"

        self.dataset = dataset
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(dataset.num_users + dataset.num_items, latent_dim)
        self.num_nodes = dataset.num_items + dataset.num_users

        # every layer is the same
        self.sheaf_conv1 = ExtendableSheafGCNLayer(latent_dim, latent_dim, self.create_operator_layers(layer_types))
        self.sheaf_conv2 = ExtendableSheafGCNLayer(latent_dim, latent_dim, self.create_operator_layers(layer_types))
        self.sheaf_conv3 = ExtendableSheafGCNLayer(latent_dim, latent_dim, self.create_operator_layers(layer_types))

        self.edge_index = self.dataset.train_edge_index
        self.adj = ExtendableSheafGCN.compute_adj_normalized(self.dataset.adjacency_matrix)
        self.init_parameters()

    @staticmethod
    def compute_adj_normalized(adjacency_matrix):
        degree = adjacency_matrix.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        diag_degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        return diag_degree_inv_sqrt @ adjacency_matrix @ diag_degree_inv_sqrt

    def create_operator_layers(self, layer_types: list[str]):
        layers = list()

        params = {
            "dimx": self.latent_dim,
            "dimy": self.latent_dim,
            "user_indices": list(range(self.dataset.num_users)),
            "item_indices": list(range(self.dataset.num_users, self.dataset.num_users + self.dataset.num_items)),
        }

        def make_layer(layer_type: str):
            match layer_type:
                case OperatorComputeLayerType.LAYER_GLOBAL:
                    return GlobalOperatorComputeLayer(**params)
                case OperatorComputeLayerType.LAYER_SINGLE_ENTITY:
                    return SingleEntityOperatorComputeLayer(**params, nsmat=64)
                case OperatorComputeLayerType.LAYER_PAIRED_ENTITIES:
                    return PairedEntityOperatorComputeLayer(**params, nsmat=64)

        for layer_type in layer_types:
            layers.append(make_layer(layer_type))

        return layers

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.sheaf_conv1.init_parameters()
        self.sheaf_conv2.init_parameters()
        self.sheaf_conv3.init_parameters()

    def forward_(self, adj_matrix):
        emb0 = self.embedding.weight
        m_u0, diff_loss, cons_loss, orth_loss = self.sheaf_conv1(adj_matrix, emb0, self.edge_index, True)
        m_u1 = self.sheaf_conv2(adj_matrix, m_u0, self.edge_index)
        out = self.sheaf_conv3(adj_matrix, m_u1, self.edge_index)

        return out, diff_loss, cons_loss, orth_loss

    def forward(self, adj_matrix):
        emb0 = self.embedding.weight
        m_u0 = self.sheaf_conv1(adj_matrix, emb0, self.edge_index)
        m_u1 = self.sheaf_conv2(adj_matrix, m_u0, self.edge_index)
        out = self.sheaf_conv3(adj_matrix, m_u1, self.edge_index)

        return emb0, out

    def training_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        embs, users_emb, pos_emb, neg_emb, loss_diff, loss_cons, loss_orth = self.encode_minibatch(users, pos_items,
                                                                                                   neg_items, self.adj)
        bpr_loss = compute_bpr_loss(users, users_emb, pos_emb, neg_emb)

        w_diff, w_orth, w_cons, w_bpr = compute_loss_weights_simple(loss_diff, loss_orth, loss_cons, bpr_loss, 1024)

        loss = w_diff * loss_diff + w_bpr * bpr_loss  # + w_orth * loss_orth + w_cons * loss_cons

        if Losses.CONSISTENCY in self.losses:
            loss += w_cons * loss_cons

        if Losses.ORTHOGONALITY in self.losses:
            loss += w_orth * loss_orth

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

    def encode_minibatch(self, users, pos_items, neg_items, adj_matrix):
        out, diff_loss, cons_loss, orth_loss = self.forward_(adj_matrix)

        return (
            out,
            out[users],
            out[pos_items],
            out[neg_items],
            diff_loss,
            cons_loss,
            orth_loss
        )
