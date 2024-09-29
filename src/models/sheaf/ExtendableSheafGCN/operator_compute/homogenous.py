import torch
from torch import nn

from .base import (
    OperatorComputeLayer,
    SheafOperators,
    LayerCompositionType,
    LayerPriority, is_zero_matrix,
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

    def is_denoisable(self):
        return True

    def compute_for_denoise(self,
                            embeddings: torch.Tensor,
                            operators: torch.Tensor) -> torch.Tensor:
        # it is always additive because this layer is the first layer in composition
        operators[:, ...] += self.operator
        return operators


class HomogenousSimpleFFNOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dim_in: int, dim_out: int, composition_type: str, nsmat: int = 64, depth: int = 6):
        super().__init__(dim_in, dim_out, composition_type)

        self.fc_smat = make_fc_transform(self.dim_in, (self.dim_in, self.dim_out), nsmat, depth=depth)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:

        operator_by_embedding = torch.reshape(self.fc_smat(embeddings), (-1, self.dim_out, self.dim_in))

        if self.composition_type == LayerCompositionType.ADDITIVE or is_zero_matrix(embeddings):
            operators.operator_uv += operator_by_embedding[u_indices, ...]
            operators.operator_vu += operator_by_embedding[v_indices, ...]
        elif self.composition_type == LayerCompositionType.MULTIPLICATIVE:
            operators.operator_uv = torch.bmm(operator_by_embedding[u_indices, ...], operators.operator_uv)
            operators.operator_vu = torch.bmm(operator_by_embedding[v_indices, ...], operators.operator_vu)
        else:
            raise Exception("unknown composition type")

        return operators

    def priority(self):
        return LayerPriority.LAYER_HOMO_SIMPLE_FFN

    def init_parameters(self):
        self.fc_smat.apply(OperatorComputeLayer.init_layer)

    def is_denoisable(self):
        return True

    def compute_for_denoise(self,
                            embeddings: torch.Tensor,
                            operators: torch.Tensor) -> torch.Tensor:
        operator_by_embedding = torch.reshape(self.fc_smat(embeddings), (-1, self.dim_out, self.dim_in))
        if self.composition_type == LayerCompositionType.ADDITIVE or is_zero_matrix(embeddings):
            operators = operators + operator_by_embedding
        elif self.composition_type == LayerCompositionType.MULTIPLICATIVE:
            operators = torch.bmm(operators, operator_by_embedding)
        else:
            raise LayerCompositionType.IncorrectCompositionException()

        return operators


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

        if self.composition_type == LayerCompositionType.ADDITIVE or is_zero_matrix(embeddings):
            operators.operator_uv += operator_uv
            operators.operator_vu += operator_vu
        elif self.composition_type == LayerCompositionType.MULTIPLICATIVE:
            operators.operator_uv = torch.bmm(operator_uv, operators.operator_uv)
            operators.operator_vu = torch.bmm(operator_vu, operators.operator_vu)
        else:
            raise LayerCompositionType.IncorrectCompositionException()

        return operators

    def priority(self):
        return LayerPriority.LAYER_HOMO_PAIRED_FFN

    def init_parameters(self):
        self.fc_smat.apply(OperatorComputeLayer.init_layer)
