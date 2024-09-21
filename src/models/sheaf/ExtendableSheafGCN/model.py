import torch
import pytorch_lightning as pl

from torch import nn
from torch_geometric.utils import to_dense_adj

from src.losses.bpr import compute_bpr_loss
from src.losses.sheaf import compute_loss_weight_paper
from src.losses import Losses


from .operator_compute_layer.base import (
    OperatorComputeLayer,
    OperatorComputeLayerType,
    LayerCompositionType,
    SheafOperators,
)
from .operator_compute_layer.heterogeneous import (
    HeterogeneousGlobalOperatorComputeLayer,
    HeterogeneousSimpleFFNOperatorComputeLayer,
)
from .operator_compute_layer.homogenous import (
    HomogenousGlobalOperatorComputeLayer,
    HomogenousSimpleFFNOperatorComputeLayer,
    HomogenousPairedFFNOperatorComputeLayer,
)


"""
This is extension over an approach implemented in EXSheafGCN. Here we use FFN over two embeddings and two global matrices
to compute linear operator. A(u, v) = FFN(u, v) + FFN(u) + U for u and v vectors (and vise versa).  
"""


class OperatorComputeLayerTrainMode:
    CONSECUTIVE = "cons"  # second, but not first, and not third yet
    INCREMENTAL = "inc"  # first and second, but not third
    SIMULTANEOUS = "sim"  # first and second and third


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
                 operator_ffn_depth: int = 6,
                 operator_train_mode: str = OperatorComputeLayerTrainMode.SIMULTANEOUS,
                 epochs_per_operator: int = 30,
                 grad_clip: float = 0.5):
        super(ExtendableSheafGCN, self).__init__()

        if layer_types is None:
            layer_types = [OperatorComputeLayerType.LAYER_HETERO_GLOBAL]

        if losses is None:
            self.losses = {Losses.BPR, Losses.DIFFUSION, Losses.ORTHOGONALITY, Losses.CONSISTENCY}
        else:
            self.losses = set(losses)

        Losses.validate(self.losses)
        OperatorComputeLayerType.validate(layer_types)

        self.dataset = dataset
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(dataset.get_num_nodes(), latent_dim)
        self.composition_type = composition_type
        self.operator_ffn_depth = operator_ffn_depth
        self.operator_train_mode = operator_train_mode
        self.epochs_per_operator = epochs_per_operator

        self.sheaf_conv = ExtendableSheafGCNLayer(
            dimx=latent_dim,
            dimy=latent_dim,
            operator_compute_layers=self.create_operator_layers(layer_types),
            operator_compute_train_mode=self.operator_train_mode,
            epochs_per_operator=self.epochs_per_operator
        )

        self.edge_index = self.dataset.train_edge_index

        # adjacency matrix can be predefined for small datasets and be missing for large datasets,
        # in this case it will be computed on forward pass
        if hasattr(self.dataset, "adjacency_matrix"):
            self.adjacency_matrix_norm = ExtendableSheafGCN.compute_adj_normalized(self.dataset.adjacency_matrix)
        else:
            self.adjacency_matrix_norm = None

        self.init_parameters()
        self.init_grad_clipping(grad_clip)

    def forward_(self, edge_index: torch.Tensor):
        adjacency_matrix_norm = self.get_normalized_adjacency_matrix(edge_index)
        emb0 = self.embedding.weight
        m_u0 = self.sheaf_conv(adjacency_matrix_norm, emb0, edge_index, False)
        m_u1 = self.sheaf_conv(adjacency_matrix_norm, m_u0, edge_index, False)
        out, diff_loss, cons_loss, orth_loss = self.sheaf_conv(adjacency_matrix_norm, m_u1, edge_index, True)

        return out, diff_loss, cons_loss, orth_loss

    def forward(self, edge_index: torch.Tensor):
        adjacency_matrix_norm = self.get_normalized_adjacency_matrix(edge_index)
        emb0 = self.embedding.weight
        m_u0 = self.sheaf_conv(adjacency_matrix_norm, emb0, edge_index)
        m_u1 = self.sheaf_conv(adjacency_matrix_norm, m_u0, edge_index)
        out = self.sheaf_conv(adjacency_matrix_norm, m_u1, edge_index)

        return emb0, out

    def get_normalized_adjacency_matrix(self, edge_index: torch.Tensor):
        if self.adjacency_matrix_norm is None:
            adjacency_matrix = torch.squeeze(to_dense_adj(edge_index))
            return ExtendableSheafGCN.compute_adj_normalized(adjacency_matrix)

        return self.adjacency_matrix_norm

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            start_nodes, pos_items, neg_items = batch
            edge_index = self.edge_index
        elif len(batch) == 4:
            start_nodes, pos_items, neg_items, edge_index = batch
        else:
            raise Exception("batch is of unknown size")

        self.update_epoch()

        embs, start_nodes_emb, pos_emb, neg_emb, loss_diff, loss_cons, loss_orth = self.encode_minibatch(start_nodes, pos_items, neg_items, edge_index)
        w_diff, w_orth, w_cons, w_bpr = compute_loss_weight_paper(loss_diff, loss_orth, loss_cons, len(start_nodes))

        bpr_loss = compute_bpr_loss(start_nodes, start_nodes_emb, pos_emb, neg_emb)
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

    def encode_minibatch(self, start_nodes, pos_items, neg_items, edge_index):
        out, diff_loss, cons_loss, orth_loss = self.forward_(edge_index)

        return (
            out,
            out[start_nodes],
            out[pos_items],
            out[neg_items],
            diff_loss,
            cons_loss,
            orth_loss
        )

    @staticmethod
    def compute_adj_normalized(adjacency_matrix):
        degree = adjacency_matrix.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        diag_degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        return diag_degree_inv_sqrt @ adjacency_matrix @ diag_degree_inv_sqrt

    def create_operator_layers(self, layer_types: list[str]):
        layers = list()

        params = dict(
            dim_in=self.latent_dim,
            dim_out=self.latent_dim,
            composition_type=self.composition_type,
        )

        for layer_type in layer_types:
            layer = (self.make_heterogeneous_layer(layer_type, **params) or
                     self.make_homogenous_layer(layer_type, **params))
            if layer is None:
                raise Exception("unknown layer type.")

            layers.append(layer)

        return layers

    def make_heterogeneous_layer(self, layer_type: str, **params):
        hetero_params = dict(
            user_indices=list(range(self.dataset.num_users)),
            item_indices=list(range(self.dataset.num_users, self.dataset.num_users + self.dataset.num_items)),
            **params
        )

        match layer_type:
            case OperatorComputeLayerType.LAYER_HETERO_GLOBAL:
                return HeterogeneousGlobalOperatorComputeLayer(**hetero_params)
            case OperatorComputeLayerType.LAYER_HETERO_SIMPLE_FFN:
                return HeterogeneousSimpleFFNOperatorComputeLayer(**dict(
                    depth=self.operator_ffn_depth,
                    nsmat=64,
                    **hetero_params,
                ))

        return None

    def make_homogenous_layer(self, layer_type: str, **params):
        match layer_type:
            case OperatorComputeLayerType.LAYER_HOMO_GLOBAL:
                return HomogenousGlobalOperatorComputeLayer(**params)
            case OperatorComputeLayerType.LAYER_HOMO_SIMPLE_FFN:
                return HomogenousSimpleFFNOperatorComputeLayer(
                    depth=self.operator_ffn_depth,
                    nsmat=64,
                    **params
                )
            case OperatorComputeLayerType.LAYER_HOMO_PAIRED_FFN:
                return HomogenousPairedFFNOperatorComputeLayer(
                    depth=self.operator_ffn_depth,
                    nsmat=64,
                    **params
                )

        return None

    def update_epoch(self):
        self.sheaf_conv.set_current_epoch(self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def init_grad_clipping(self, clip_value):
        for param in self.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.sheaf_conv.init_parameters()
