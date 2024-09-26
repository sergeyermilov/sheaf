import dataclasses

import torch
from torch import nn


class OperatorComputeLayerType:
    LAYER_HOMO_GLOBAL = "homo_global"
    LAYER_HOMO_SIMPLE_FFN = "homo_simple_ffn"
    LAYER_HOMO_PAIRED_FFN = "homo_paired_ffn"
    LAYER_HETERO_GLOBAL = "hetero_global"
    LAYER_HETERO_SIMPLE_FFN = "hetero_simple_ffn"

    @staticmethod
    def validate(layers):
        valid_layers = {
            OperatorComputeLayerType.LAYER_HOMO_GLOBAL,
            OperatorComputeLayerType.LAYER_HOMO_SIMPLE_FFN,
            OperatorComputeLayerType.LAYER_HOMO_PAIRED_FFN,
            OperatorComputeLayerType.LAYER_HETERO_GLOBAL,
            OperatorComputeLayerType.LAYER_HETERO_SIMPLE_FFN
        }

        assert layers and all([loss in valid_layers for loss in layers]), "none or invalid layers"

    @staticmethod
    def is_homo(layers):
        valid_layers = {
            OperatorComputeLayerType.LAYER_HOMO_GLOBAL,
            OperatorComputeLayerType.LAYER_HOMO_SIMPLE_FFN,
            OperatorComputeLayerType.LAYER_HOMO_PAIRED_FFN,
        }

        assert layers and all([loss in valid_layers for loss in layers]), "no homo"

    @staticmethod
    def is_hetero(layers):
        valid_layers = {
            OperatorComputeLayerType.LAYER_HETERO_GLOBAL,
            OperatorComputeLayerType.LAYER_HETERO_SIMPLE_FFN
        }

        assert layers and all([loss in valid_layers for loss in layers]), "no hetero"


class LayerCompositionType:
    MULTIPLICATIVE = "mult"
    ADDITIVE = "add"

    @staticmethod
    def validate(composition_type: str):
        if composition_type not in {
            LayerCompositionType.MULTIPLICATIVE,
            LayerCompositionType.ADDITIVE
        }:
            raise LayerCompositionType.IncorrectCompositionException()

    class IncorrectCompositionException(Exception):
        def __init__(self):
            super().__init__("Incorrect composition")


class LayerPriority:
    LAYER_HOMO_GLOBAL=0
    LAYER_HOMO_SIMPLE_FFN=2
    LAYER_HOMO_PAIRED_FFN=4
    LAYER_HETERO_GLOBAL=1
    LAYER_HETERO_SIMPLE_FFN=3


@dataclasses.dataclass
class SheafOperators:
    operator_uv: torch.Tensor  # A(u, v)
    operator_vu: torch.Tensor  # A(v, u)


class OperatorComputeLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, composition_type: str):
        super(OperatorComputeLayer, self).__init__()

        LayerCompositionType.validate(composition_type)

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.composition_type = composition_type

    def assert_operators(self, operator: torch.Tensor):
        assert (operator.shape[-2], operator.shape[-1]) == (self.dim_out, self.dim_in), "invalid shape"

    def forward(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:
        self.assert_operators(operators.operator_uv)
        self.assert_operators(operators.operator_vu)

        OperatorComputeLayer.assert_indices(u_indices, embeddings)
        OperatorComputeLayer.assert_indices(v_indices, embeddings)

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

    # Use \hat{x} = A^T * A * x instead of message passing,
    # only available for no-/single- feature computers
    def compute_for_denoise(self,
                            embeddings: torch.Tensor,
                            operators: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def is_denoisable(self):
        return False

    @staticmethod
    def init_layer(layer: nn.Module):
        if layer is nn.Linear:
            nn.init.xavier_uniform(layer.weight)

    @staticmethod
    def assert_indices(indices: torch.Tensor, embeddings: torch.Tensor):
        assert torch.max(indices) < embeddings.shape[0], "invalid indices"


def is_zero_matrix(embeddings: torch.Tensor):
    return torch.allclose(embeddings.sum(), torch.tensor(0))
