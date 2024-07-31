import torch
import torch.nn as nn
from unittest import TestCase

from src.models.sheaf.ExtendableSheafGCN import (
    SheafOperators,
    GlobalOperatorComputeLayer,
    SingleEntityOperatorComputeLayer,
    PairedEntityOperatorComputeLayer,
)


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class TestExtendableSheafGCN(TestCase):
    def __init__(self, *args):
        super().__init__(*args)

        self.dimx = 40
        self.dimy = 40
        self.user_indices = list(range(0, 3))
        self.item_indices = list(range(3, 5))

        self.edge_index = torch.tensor([
            [0, 3],
            [1, 3],
            [1, 4],
            [2, 3],

            [3, 0],
            [3, 1],
            [4, 1],
            [3, 2],
        ], dtype=torch.int32)

        self.embeddings = torch.tensor([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ], dtype=torch.int32)

    def make_sheaf_operators(self):
        return SheafOperators(
            torch.zeros((self.edge_index.shape[0], self.dimy, self.dimx), requires_grad=False),
            torch.zeros((self.edge_index.shape[0], self.dimy, self.dimx), requires_grad=False)
        )

    def assert_operators(self, sheaf_operators):
        assert sheaf_operators.operator_uv[:4, 0, 0].min() == 1, \
            "Incorrect user matrix A(u, v) element."
        assert sheaf_operators.operator_uv[4:, 0, 0].min() == 2, \
            "Incorrect item matrix A(u, v) element."

        assert sheaf_operators.operator_vu[:4, 0, 0].min() == 2, \
            "Incorrect item matrix A(v, u) element."
        assert sheaf_operators.operator_vu[4:, 0, 0].min() == 1, \
            "Incorrect user matrix A(v, u) element."

    def test_global_layer(self):
        global_layer = GlobalOperatorComputeLayer(
            dimx=self.dimx,
            dimy=self.dimy,
            user_indices=self.user_indices,
            item_indices=self.item_indices
        )

        global_layer.user_operator.data.fill_(1.0)
        global_layer.item_operator.data.fill_(2.0)

        sheaf_operators: SheafOperators = global_layer.forward(
            u_indices=self.edge_index[:, 0].view(-1),
            v_indices=self.edge_index[:, 1].view(-1),
            embeddings=self.embeddings,
            operators=self.make_sheaf_operators()
        )

        self.assert_operators(sheaf_operators)

    def test_single_layer(self):
        def compute_matrix(embeddings: torch.Tensor) -> torch.Tensor:
            mask = torch.isin(
                embeddings.argmax(dim=1),
                torch.tensor(self.user_indices)
            )

            matrices = torch.ones((embeddings.shape[0], self.dimx, self.dimy))

            matrices[mask, ...] = 1
            matrices[~mask, ...] = 2

            return matrices

        single_layer = SingleEntityOperatorComputeLayer(
            dimx=self.dimx,
            dimy=self.dimy,
            user_indices=self.user_indices,
            item_indices=self.item_indices
        )
        single_layer.fc_smat = LambdaModule(compute_matrix)

        sheaf_operators: SheafOperators = single_layer.forward(
            u_indices=self.edge_index[:, 0].view(-1),
            v_indices=self.edge_index[:, 1].view(-1),
            embeddings=self.embeddings,
            operators=self.make_sheaf_operators()
        )

        self.assert_operators(sheaf_operators)

    def test_paired_layer(self):
        def compute_matrix(embeddings: torch.Tensor) -> torch.Tensor:
            mask = torch.isin(
                embeddings[:, :embeddings.shape[1] // 2].argmax(dim=1),
                torch.tensor(self.user_indices)
            )

            matrices = torch.ones((embeddings.shape[0], self.dimx, self.dimy))

            matrices[mask, ...] = 1
            matrices[~mask, ...] = 2

            return matrices

        paired_layer = PairedEntityOperatorComputeLayer(
            dimx=self.dimx,
            dimy=self.dimy,
            user_indices=self.user_indices,
            item_indices=self.item_indices
        )
        paired_layer.fc_smat = LambdaModule(compute_matrix)

        sheaf_operators: SheafOperators = paired_layer.forward(
            u_indices=self.edge_index[:, 0].view(-1),
            v_indices=self.edge_index[:, 1].view(-1),
            embeddings=self.embeddings,
            operators=self.make_sheaf_operators()
        )

        self.assert_operators(sheaf_operators)
