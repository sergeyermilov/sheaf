import dataclasses

import torch
from torch import nn


class LayerCompositionType:
    MULTIPLICATIVE = "mult"
    ADDITIVE = "add"


class OperatorComputeLayerType:
    LAYER_GLOBAL = "global"
    LAYER_SINGLE_ENTITY = "single"
    LAYER_SINGLE_ENTITY_DISTINCT = "single_distinct"
    LAYER_PAIRED_ENTITIES = "paired"

    @staticmethod
    def validate(layers):
        valid_layers = {
            OperatorComputeLayerType.LAYER_GLOBAL,
            OperatorComputeLayerType.LAYER_SINGLE_ENTITY,
            OperatorComputeLayerType.LAYER_SINGLE_ENTITY_DISTINCT,
            OperatorComputeLayerType.LAYER_PAIRED_ENTITIES
        }

        assert layers and all([loss in valid_layers for loss in layers]), "none or invalid layers"


def make_fc_transform(inpt: int, outpt: tuple[int, int], nsmat: int, depth: int = 6, dropout_proba: float = 0):
    assert len(outpt) == 2, "incorrect output dim"

    layers = [nn.Linear(inpt, nsmat), nn.ReLU()]

    for i in range(depth):
        layers.append(nn.Linear(nsmat, nsmat))
        if dropout_proba != 0:
            layers.append(nn.Dropout(dropout_proba))
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
