import torch
import pytorch_lightning as pl

from torch import nn
from torch_geometric.utils import to_dense_adj

from src.losses.bpr import compute_bpr_loss
from src.losses.sheaf import compute_loss_weight_paper
from src.losses import Losses


from .operator_compute.base import (
    OperatorComputeLayerType,
    LayerCompositionType,
)
from .operator_compute.heterogeneous import (
    HeterogeneousGlobalOperatorComputeLayer,
    HeterogeneousSimpleFFNOperatorComputeLayer,
)
from .operator_compute.homogenous import (
    HomogenousGlobalOperatorComputeLayer,
    HomogenousSimpleFFNOperatorComputeLayer,
    HomogenousPairedFFNOperatorComputeLayer,
)
from .sheaf_layer import (
    OperatorComputeLayerTrainMode,
    ExtendableSheafGCNLayer
)

"""
This is extension over an approach implemented in EXSheafGCN. Here we use FFN over two embeddings and two global matrices
to compute linear operator. A(u, v) = FFN(u, v) + FFN(u) + U for u and v vectors (and vise versa).  
"""


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

    def is_denoisable(self):
        return self.sheaf_conv.is_denoisable()

    def get_denoised_embeddings(self):
        emb0 = self.embedding.weight
        return self.sheaf_conv.denoise(emb0)

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

    @staticmethod
    def compute_adj_normalized(adjacency_matrix):
        degree = adjacency_matrix.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        diag_degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        return diag_degree_inv_sqrt @ adjacency_matrix @ diag_degree_inv_sqrt
