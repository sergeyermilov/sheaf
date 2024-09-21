import torch
from torch import nn

from .base import (
    OperatorComputeLayer,
    SheafOperators,
    LayerCompositionType,
    LayerPriority,
)
from .utils import make_fc_transform


class HomogenousGlobalOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dim_in: int, dim_out: int, composition_type: str):
        super().__init__(dim_in, dim_out, composition_type)

        self.operator = nn.Parameter(torch.zeros((self.dim_out, self.dim_in)), requires_grad=True)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:
        # it is always additive becausse this layer is the first layer in composition
        operators.operator_uv[:, ...] += self.operator
        operators.operator_vu[:, ...] += self.operator
        return operators

    def priority(self):
        return LayerPriority.LAYER_HOMO_GLOBAL

    def init_parameters(self):
        nn.init.xavier_uniform(self.operator.data)


class HomogenousSimpleFFNOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dim_in: int, dim_out: int, composition_type: str, nsmat: int = 64, depth: int = 6):
        super().__init__(dim_in, dim_out, composition_type)

        # maybe create two selarate FFNs for user and item nodes?
        self.fc_smat = make_fc_transform(self.dim_in, (self.dim_in, self.dim_out), nsmat, depth=depth)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:

        operator_by_embedding = torch.reshape(self.fc_smat(embeddings), (-1, self.dim_out, self.dim_in))

        if self.composition_type == LayerCompositionType.ADDITIVE or torch.allclose(embeddings.sum(), torch.tensor(0)):
            operators.operator_uv += operator_by_embedding[u_indices, ...]
            operators.operator_vu += operator_by_embedding[v_indices, ...]
        else:
            operators.operator_uv = torch.bmm(operator_by_embedding[u_indices, ...], operators.operator_uv)
            operators.operator_vu = torch.bmm(operator_by_embedding[v_indices, ...], operators.operator_vu)

        return operators

    def priority(self):
        return LayerPriority.LAYER_HOMO_SIMPLE_FFN

    def init_parameters(self):
        self.fc_smat.apply(OperatorComputeLayer.init_layer)


class HomogenousPairedFFNOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dim_in: int, dim_out: int, composition_type: str, nsmat: int = 32, depth: int = 6):
        super().__init__(dim_in, dim_out, composition_type)

        self.dim_in = dim_in
        self.dim_out = dim_out

        # maybe create two selarate FFNs for user and item nodes?
        self.fc_smat = make_fc_transform(self.dim_in * 2, (self.dim_in, self.dim_out), nsmat, depth=depth)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:

        u_embeddings = embeddings[u_indices, ...]
        v_embeddings = embeddings[v_indices, ...]

        combined_embeddings_uv = torch.concat([u_embeddings, v_embeddings], dim=-1)
        combined_embeddings_vu = torch.concat([v_embeddings, u_embeddings], dim=-1)

        operator_uv = torch.reshape(self.fc_smat(combined_embeddings_uv), (-1, self.dim_out, self.dim_in))
        operator_vu = torch.reshape(self.fc_smat(combined_embeddings_vu), (-1, self.dim_out, self.dim_in))

        if self.composition_type == LayerCompositionType.ADDITIVE or torch.allclose(embeddings.sum(), torch.tensor(0)):
            operators.operator_uv += operator_uv
            operators.operator_vu += operator_vu
        else:
            operators.operator_uv = torch.bmm(operator_uv, operators.operator_uv)
            operators.operator_vu = torch.bmm(operator_vu, operators.operator_vu)

        return operators

    def priority(self):
        return LayerPriority.LAYER_HOMO_PAIRED_FFN

    def init_parameters(self):
        self.fc_smat.apply(OperatorComputeLayer.init_layer)
