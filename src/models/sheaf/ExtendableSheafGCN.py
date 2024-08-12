import dataclasses

import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric.utils import dropout_edge

from src.losses.bpr import compute_bpr_loss, compute_loss_weights_simple, compute_loss_weight_paper

"""
This is extension over an approach implemented in EXSheafGCN. Here we use FFN over two embeddings and two global matrices
to compute linear operator. A(u, v) = FFN(u, v) + FFN(u) + U for u and v vectors (and vise versa).  
"""


class Losses:
    ORTHOGONALITY = "orth"
    CONSISTENCY = "cons"


class OperatorComputeLayerType:
    LAYER_GLOBAL = "global"
    LAYER_SINGLE_ENTITY = "single"
    LAYER_SINGLE_ENTITY_DISTINCT = "single_distinct"
    LAYER_PAIRED_ENTITIES = "paired"


class LayerCompositionType:
    MULTIPLICATIVE = "mult"
    ADDITIVE = "add"


class OperatorComputeLayerTrainMode:
    CONSECUTIVE = "cons"  # second, but not first, and not third yet
    INCREMENTAL = "inc"  # first and second, but not third
    SIMULTANEOUS = "sim"  # first and second and third


def make_fc_transform(inpt: int, outpt: tuple[int, int], nsmat: int, depth: int = 6):
    assert len(outpt) == 2, "incorrect output dim"

    layers = [nn.Linear(inpt, nsmat), nn.ReLU()]

    for i in range(depth):
        layers.append(nn.Linear(nsmat, nsmat))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(nsmat, outpt[0] * outpt[1]))
    return nn.Sequential(*layers)


@dataclasses.dataclass
class SheafOperators:
    operator_uv: torch.Tensor  # A(u, v)
    operator_vu: torch.Tensor  # A(v, u)


class OperatorComputeLayer(nn.Module):
    def __init__(self, dimx: int, dimy: int, user_indices: list[int], item_indices: list[int], composition_type: str):
        super(OperatorComputeLayer, self).__init__()

        assert composition_type in {LayerCompositionType.ADDITIVE, LayerCompositionType.MULTIPLICATIVE}, "incorrect composition type"

        self.dimx = dimx
        self.dimy = dimy

        self.composition_type = composition_type

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

    def priority(self):
        raise NotImplementedError()

    @staticmethod
    def init_layer(layer):
        if layer is nn.Linear:
            nn.init.xavier_uniform(layer.weight)


class GlobalOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dimx: int, dimy: int, user_indices: list[int], item_indices: list[int], composition_type: str):
        super(GlobalOperatorComputeLayer, self).__init__(dimx, dimy, user_indices, item_indices, composition_type)

        self.user_operator = nn.Parameter(torch.zeros((self.dimy, self.dimx)), requires_grad=True)
        self.item_operator = nn.Parameter(torch.zeros((self.dimy, self.dimx)), requires_grad=True)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:
        # it is always additive becausse this layer is the first layer in composition
        operators.operator_uv[torch.isin(u_indices, self.user_indices), ...] += self.user_operator
        operators.operator_uv[torch.isin(u_indices, self.item_indices), ...] += self.item_operator
        operators.operator_vu[torch.isin(v_indices, self.user_indices), ...] += self.user_operator
        operators.operator_vu[torch.isin(v_indices, self.item_indices), ...] += self.item_operator
        return operators

    def priority(self):
        return 1

    def init_parameters(self):
        nn.init.xavier_uniform(self.user_operator.data)
        nn.init.xavier_uniform(self.item_operator.data)


class SingleEntityOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dimx: int, dimy: int, user_indices: list[int], item_indices: list[int], composition_type: str, nsmat: int = 64, depth: int = 6):
        super(SingleEntityOperatorComputeLayer, self).__init__(dimx, dimy, user_indices, item_indices, composition_type)

        # maybe create two selarate FFNs for user and item nodes?
        self.fc_smat = make_fc_transform(self.dimx, (self.dimx, self.dimy), nsmat, depth=depth)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:

        operator_by_embedding = torch.reshape(self.fc_smat(embeddings), (-1, self.dimy, self.dimx))

        if self.composition_type == LayerCompositionType.ADDITIVE or torch.allclose(embeddings.sum(), torch.tensor(0)):
            operators.operator_uv += operator_by_embedding[u_indices, ...]
            operators.operator_vu += operator_by_embedding[v_indices, ...]
        else:
            operators.operator_uv = torch.bmm(operator_by_embedding[u_indices, ...], operators.operator_uv)
            operators.operator_vu = torch.bmm(operator_by_embedding[v_indices, ...], operators.operator_vu)

        return operators

    def priority(self):
        return 2

    def init_parameters(self):
        self.fc_smat.apply(OperatorComputeLayer.init_layer)


class SingleEntityDistinctOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dimx: int, dimy: int, user_indices: list[int], item_indices: list[int], composition_type: str, nsmat: int = 64, depth: int = 6):
        super(SingleEntityDistinctOperatorComputeLayer, self).__init__(dimx, dimy, user_indices, item_indices, composition_type)

        # maybe create two selarate FFNs for user and item nodes?
        self.fc_smat_user = make_fc_transform(self.dimx, (self.dimx, self.dimy), nsmat, depth=depth)
        self.fc_smat_item = make_fc_transform(self.dimx, (self.dimx, self.dimy), nsmat, depth=depth)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:

        operator_by_embedding_user = torch.reshape(self.fc_smat_user(embeddings), (-1, self.dimy, self.dimx))
        operator_by_embedding_item = torch.reshape(self.fc_smat_item(embeddings), (-1, self.dimy, self.dimx))

        u_user_mask = torch.isin(u_indices, self.user_indices)
        u_item_mask = torch.isin(u_indices, self.item_indices)
        v_user_mask = torch.isin(v_indices, self.user_indices)
        v_item_mask = torch.isin(v_indices, self.item_indices)

        if self.composition_type == LayerCompositionType.ADDITIVE or torch.allclose(embeddings.sum(), torch.tensor(0)):
            operators.operator_uv[u_user_mask, ...] += operator_by_embedding_user[u_indices[u_user_mask], ...]
            operators.operator_uv[u_item_mask, ...] += operator_by_embedding_item[u_indices[u_item_mask], ...]
            operators.operator_vu[v_user_mask, ...] += operator_by_embedding_user[v_indices[v_user_mask], ...]
            operators.operator_vu[v_item_mask, ...] += operator_by_embedding_item[v_indices[v_item_mask], ...]
        else:
            operators.operator_uv[u_user_mask, ...] = torch.bmm(operator_by_embedding_user[u_indices[u_user_mask], ...], operators.operator_uv[u_user_mask, ...])
            operators.operator_uv[u_item_mask, ...] = torch.bmm(operator_by_embedding_item[u_indices[u_item_mask], ...], operators.operator_uv[u_item_mask, ...])
            operators.operator_vu[v_user_mask, ...] = torch.bmm(operator_by_embedding_user[v_indices[v_user_mask], ...], operators.operator_vu[v_user_mask, ...])
            operators.operator_vu[v_item_mask, ...] = torch.bmm(operator_by_embedding_item[v_indices[v_item_mask], ...], operators.operator_vu[v_item_mask, ...])

        return operators

    def priority(self):
        return 3

    def init_parameters(self):
        self.fc_smat_user.apply(OperatorComputeLayer.init_layer)
        self.fc_smat_item.apply(OperatorComputeLayer.init_layer)


class PairedEntityOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dimx: int, dimy: int, user_indices: list[int], item_indices: list[int], composition_type: str, nsmat: int = 32, depth: int = 6):
        super(PairedEntityOperatorComputeLayer, self).__init__(dimx, dimy, user_indices, item_indices, composition_type)

        self.dimx = dimx
        self.dimy = dimy

        # maybe create two selarate FFNs for user and item nodes?
        self.fc_smat = make_fc_transform(self.dimx * 2, (self.dimx, self.dimy), nsmat, depth=depth)

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

        if self.composition_type == LayerCompositionType.ADDITIVE or torch.allclose(embeddings.sum(), torch.tensor(0)):
            operators.operator_uv += operator_uv
            operators.operator_vu += operator_vu
        else:
            operators.operator_uv = torch.bmm(operator_uv, operators.operator_uv)
            operators.operator_vu = torch.bmm(operator_vu, operators.operator_vu)

        return operators

    def priority(self):
        return 4

    def init_parameters(self):
        self.fc_smat.apply(OperatorComputeLayer.init_layer)


class ExtendableSheafGCNLayer(nn.Module):
    def __init__(self,
                 dimx: int,
                 dimy: int,
                 operator_compute_layers: list[OperatorComputeLayer],
                 operator_compute_train_mode: str = OperatorComputeLayerTrainMode.SIMULTANEOUS,
                 epochs_per_operator: int = None):
        super(ExtendableSheafGCNLayer, self).__init__()
        self.dimx = dimx
        self.dimy = dimy
        self.operator_compute_layers = nn.ModuleList(operator_compute_layers)
        self.epochs_per_operator = epochs_per_operator
        self.train_mode = operator_compute_train_mode

        self.orth_eye = torch.eye(self.dimy).unsqueeze(0)
        self.current_epoch = 0

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

    def compute_layer(self, layer_ix, layer, **params):
        expected_layer_ix = self.current_epoch % self.epochs_per_operator

        def infer_no_grad():
            with torch.inference_mode():
                return layer(**params)

        def infer():
            return layer(**params)

        match self.train_mode:
            case OperatorComputeLayerTrainMode.SIMULTANEOUS:
                return infer()
            case OperatorComputeLayerTrainMode.INCREMENTAL:
                if layer_ix <= expected_layer_ix:
                    return infer()
            case OperatorComputeLayerTrainMode.CONSECUTIVE:
                if layer_ix < expected_layer_ix:
                    return infer_no_grad()

                if layer_ix == expected_layer_ix:
                    return infer()

        return None

    def forward(self, adj_matrix, embeddings, edge_index, compute_losses: bool = False):
        u_indices = edge_index[0, :]
        v_indices = edge_index[1, :]

        sheaf_operators = SheafOperators(
            torch.zeros((edge_index.shape[1], self.dimy, self.dimx), requires_grad=False),
            torch.zeros((edge_index.shape[1], self.dimy, self.dimx), requires_grad=False)
        )

        for layer_ix, operator_compute_layer in enumerate(sorted(self.operator_compute_layers, key=lambda x: x.priority())):
            sheaf_operators_updated = self.compute_layer(
                layer_ix, operator_compute_layer,
                operators=sheaf_operators, embeddings=embeddings, u_indices=u_indices, v_indices=v_indices
            )

            if sheaf_operators_updated is not None:
                sheaf_operators = sheaf_operators_updated

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

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch


class ExtendableSheafGCN(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset,
                 layer_types: list[str] = None,
                 losses: list[str] = None,
                 composition_type: str = LayerCompositionType.ADDITIVE,
                 sample_share: float = 1.0,
                 operator_ffn_depth: int = 6,
                 operator_train_mode: str = OperatorComputeLayerTrainMode.SIMULTANEOUS,
                 epochs_per_operator: int = 30):
        super(ExtendableSheafGCN, self).__init__()

        if layer_types is None:
            layer_types = [OperatorComputeLayerType.LAYER_SINGLE_ENTITY]

        if losses is None:
            self.losses = {Losses.ORTHOGONALITY, Losses.CONSISTENCY}
        else:
            self.losses = set(losses)

        assert all([loss in {Losses.ORTHOGONALITY, Losses.CONSISTENCY} for loss in self.losses]), "unknown loss type"
        assert layer_types, "layers may not be empty"

        self.dataset = dataset
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(dataset.num_users + dataset.num_items, latent_dim)
        self.num_nodes = dataset.num_items + dataset.num_users
        self.composition_type = composition_type
        self.sample_share = sample_share
        self.operator_ffn_depth = operator_ffn_depth
        self.operator_train_mode = operator_train_mode
        self.epochs_per_operator = epochs_per_operator

        # every layer is the same
        self.sheaf_conv1 = ExtendableSheafGCNLayer(latent_dim, latent_dim, self.create_operator_layers(layer_types), self.operator_train_mode, self.epochs_per_operator)
        # self.sheaf_conv2 = ExtendableSheafGCNLayer(latent_dim, latent_dim, self.create_operator_layers(layer_types), self.operator_train_mode, self.epochs_per_operator)
        # self.sheaf_conv3 = ExtendableSheafGCNLayer(latent_dim, latent_dim, self.create_operator_layers(layer_types), self.operator_train_mode, self.epochs_per_operator)

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
            "composition_type": self.composition_type,
        }

        def make_layer(layer_type: str):
            match layer_type:
                case OperatorComputeLayerType.LAYER_GLOBAL:
                    return GlobalOperatorComputeLayer(**params)
                case OperatorComputeLayerType.LAYER_SINGLE_ENTITY:
                    params.update({"depth": self.operator_ffn_depth})
                    return SingleEntityOperatorComputeLayer(**params, nsmat=64)
                case OperatorComputeLayerType.LAYER_PAIRED_ENTITIES:
                    params.update({"depth": self.operator_ffn_depth})
                    return PairedEntityOperatorComputeLayer(**params, nsmat=64)
                case OperatorComputeLayerType.LAYER_SINGLE_ENTITY_DISTINCT:
                    params.update({"depth": self.operator_ffn_depth})
                    return SingleEntityDistinctOperatorComputeLayer(**params, nsmat=64)

        for layer_type in layer_types:
            layers.append(make_layer(layer_type))

        return layers

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.sheaf_conv1.init_parameters()
        # self.sheaf_conv2.init_parameters()
        # self.sheaf_conv3.init_parameters()

    def forward_(self, edge_index):
        emb0 = self.embedding.weight
        m_u0 = self.sheaf_conv1(self.adj, emb0, edge_index, False)
        m_u1 = self.sheaf_conv1(self.adj, m_u0, edge_index, False)
        out, diff_loss, cons_loss, orth_loss = self.sheaf_conv1(self.adj, m_u1, edge_index, True)

        return out, diff_loss, cons_loss, orth_loss

    def forward(self, edge_index):
        emb0 = self.embedding.weight
        m_u0 = self.sheaf_conv1(self.adj, emb0, edge_index)
        m_u1 = self.sheaf_conv1(self.adj, m_u0, edge_index)
        out = self.sheaf_conv1(self.adj, m_u1, edge_index)

        return emb0, out

    def training_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        if self.sample_share == 1.0:
            edge_index = self.edge_index
        else:
            edge_index, _ = dropout_edge(
                self.edge_index,
                p=self.sample_share
            )

        self.update_epoch()

        embs, users_emb, pos_emb, neg_emb, loss_diff, loss_cons, loss_orth = self.encode_minibatch(users, pos_items, neg_items, edge_index)
        w_diff, w_orth, w_cons, w_bpr = compute_loss_weight_paper(loss_diff, loss_orth, loss_cons, len(users))

        bpr_loss = compute_bpr_loss(users, users_emb, pos_emb, neg_emb)
        loss = w_diff * loss_diff + w_bpr * bpr_loss

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

    def update_epoch(self):
        self.sheaf_conv1.set_current_epoch(self.current_epoch)
        # self.sheaf_conv2.set_current_epoch(self.current_epoch)
        # self.sheaf_conv3.set_current_epoch(self.current_epoch)

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        out, diff_loss, cons_loss, orth_loss = self.forward_(edge_index)

        return (
            out,
            out[users],
            out[pos_items],
            out[neg_items],
            diff_loss,
            cons_loss,
            orth_loss
        )
