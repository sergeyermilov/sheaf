import torch
from torch import nn

from .base import OperatorComputeLayer, SheafOperators, LayerCompositionType, LayerPriority
from .utils import make_fc_transform


class HeterogeneousOperatorComputeLayer(OperatorComputeLayer):
    def __init__(self, dim_in: int, dim_out: int, composition_type: str, user_indices: list[int], item_indices: list[int]):
        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            composition_type=composition_type
        )

        self.user_indices = torch.tensor(user_indices)
        self.item_indices = torch.tensor(item_indices)


class HeterogeneousGlobalOperatorComputeLayer(HeterogeneousOperatorComputeLayer):
    def __init__(self, dim_in: int, dim_out: int, user_indices: list[int], item_indices: list[int], composition_type: str):
        super().__init__(dim_in, dim_out, composition_type, user_indices, item_indices)

        self.user_operator = nn.Parameter(torch.zeros((self.dim_out, self.dim_in)), requires_grad=True)
        self.item_operator = nn.Parameter(torch.zeros((self.dim_out, self.dim_in)), requires_grad=True)

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
        return LayerPriority.LAYER_HETERO_GLOBAL

    def init_parameters(self):
        nn.init.xavier_uniform(self.user_operator.data)
        nn.init.xavier_uniform(self.item_operator.data)


class HeterogeneousSimpleFFNOperatorComputeLayer(HeterogeneousOperatorComputeLayer):
    def __init__(self, dim_in: int, dim_out: int, user_indices: list[int], item_indices: list[int], composition_type: str, nsmat: int = 64, depth: int = 6):
        super().__init__(dim_in, dim_out, composition_type, user_indices, item_indices)

        # maybe create two selarate FFNs for user and item nodes?
        self.fc_smat_user = make_fc_transform(self.dim_in, (self.dim_in, self.dim_out), nsmat, depth=depth)
        self.fc_smat_item = make_fc_transform(self.dim_in, (self.dim_in, self.dim_out), nsmat, depth=depth)

    def compute(self,
                operators: SheafOperators,
                embeddings: torch.Tensor,
                u_indices: torch.Tensor,
                v_indices: torch.Tensor) -> SheafOperators:

        operator_by_embedding_user = torch.reshape(self.fc_smat_user(embeddings), (-1, self.dim_out, self.dim_in))
        operator_by_embedding_item = torch.reshape(self.fc_smat_item(embeddings), (-1, self.dim_out, self.dim_in))

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
        return LayerPriority.LAYER_HETERO_SIMPLE_FFN

    def init_parameters(self):
        self.fc_smat_user.apply(OperatorComputeLayer.init_layer)
        self.fc_smat_item.apply(OperatorComputeLayer.init_layer)

