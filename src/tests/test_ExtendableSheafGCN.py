import torch
import torch.nn as nn
from unittest import TestCase

from src.models.sheaf.ExtendableSheafGCN import (
    SheafOperators,
    GlobalOperatorComputeLayer,
    SingleEntityOperatorComputeLayer,
    PairedEntityOperatorComputeLayer,
    ExtendableSheafGCNLayer,
    LayerCompositionType,
    OperatorComputeLayerTrainMode,
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
        ], dtype=torch.int64)

        self.embeddings = torch.tensor([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ], dtype=torch.float32)

        self.adj_matrix = torch.tensor([
            [0, 0, 0, 1, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 3, 0],
            [1, 2, 3, 0, 0],
            [0, 2, 0, 0, 0],
        ], dtype=torch.float32)

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
            item_indices=self.item_indices,
            composition_type=LayerCompositionType.ADDITIVE
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
            item_indices=self.item_indices,
            composition_type=LayerCompositionType.ADDITIVE
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
            item_indices=self.item_indices,
            composition_type=LayerCompositionType.ADDITIVE
        )
        paired_layer.fc_smat = LambdaModule(compute_matrix)

        sheaf_operators: SheafOperators = paired_layer.forward(
            u_indices=self.edge_index[:, 0].view(-1),
            v_indices=self.edge_index[:, 1].view(-1),
            embeddings=self.embeddings,
            operators=self.make_sheaf_operators()
        )

        self.assert_operators(sheaf_operators)

    def test_layer_ordering(self):
        global_layer = GlobalOperatorComputeLayer(
            dimx=self.dimx,
            dimy=self.dimy,
            user_indices=self.user_indices,
            item_indices=self.item_indices,
            composition_type=LayerCompositionType.ADDITIVE
        )

        paired_layer = PairedEntityOperatorComputeLayer(
            dimx=self.dimx,
            dimy=self.dimy,
            user_indices=self.user_indices,
            item_indices=self.item_indices,
            composition_type=LayerCompositionType.ADDITIVE
        )

        single_layer = SingleEntityOperatorComputeLayer(
            dimx=self.dimx,
            dimy=self.dimy,
            user_indices=self.user_indices,
            item_indices=self.item_indices,
            composition_type=LayerCompositionType.ADDITIVE
        )

        sorted_layers = sorted([global_layer, single_layer, paired_layer], key=lambda layer: layer.priority())

        assert sorted_layers[0] == global_layer, "incorrect ordering"
        assert sorted_layers[1] == single_layer, "incorrect ordering"
        assert sorted_layers[2] == paired_layer, "incorrect ordering"

    def test_compute_sheaf(self):
        A_uv_t = torch.tensor([
            [1, 2, 1],
            [4, 0, 1],
            [7, 8, 1]
        ], dtype=torch.float32)

        A_vu = A_uv_t.inverse().unsqueeze(0)
        A_uv_t = A_uv_t.unsqueeze(0)
        embeddings = torch.rand((1, 3))

        result = ExtendableSheafGCNLayer.compute_sheaf(A_uv_t, A_vu, embeddings, [0])
        # sheaf should be identity transformation
        assert torch.allclose(embeddings, result, atol=1e-6), "Incorrect result"

    def test_scale_sheaf(self):
        # compute c_v = w(v,u) * h_v
        embeddings = torch.ones((self.edge_index.shape[0], 3), dtype=torch.float32)
        result = ExtendableSheafGCNLayer.scale_sheaf(
            self.adj_matrix, self.edge_index[:, 0], self.edge_index[:, 1], embeddings
        )
        actual, _ = torch.max(result, dim=1)
        expected = torch.tensor([1, 2, 2, 3, 1, 2, 2, 3], dtype=torch.float32)
        assert torch.allclose(actual, expected, atol=1e-6), "Incorrect result"

    def test_compute_message(self):
        sheafs = torch.ones((self.edge_index.shape[0], self.embeddings.shape[1]), dtype=torch.float32)
        messages = ExtendableSheafGCNLayer.compute_message(self.embeddings, self.edge_index[:, 0], sheafs)
        actual, _ = torch.max(messages, dim=1)
        expected = torch.tensor([1, 2, 1, 3, 1], dtype=torch.float32)
        assert torch.allclose(actual, expected, atol=1e-6), "Incorrect result"

    def test_diff_loss(self):
        gaus = torch.randn(6, 6)
        svd = torch.linalg.svd(gaus)
        orth = svd[0] @ svd[2]

        messages = torch.clone(orth)
        embeddings = torch.clone(orth)

        embeddings[:, :] *= 2

        actual = ExtendableSheafGCNLayer.compute_diff_loss(messages, embeddings)

        assert torch.allclose(actual, torch.tensor(6. / 36), atol=1e-6), "Incorrect result"

    def test_cons_loss(self):
        # computation is straight forward but test is not, maybe implement it in future
        pass

    def test_orth_loss(self):
        gaus = torch.randn(6, 6)
        svd = torch.linalg.svd(gaus)
        orth = svd[0] @ svd[2]
        eye = torch.eye(orth.shape[0])

        A = orth.unsqueeze(0)
        A_t = A.swapaxes(-1, -2)

        actual = ExtendableSheafGCNLayer.compute_orth_loss(A, 2 * A_t, eye)

        assert torch.allclose(actual, torch.tensor(6.), atol=1e-6), "Incorrect result"

    def test_compute_layer(self):
        order = [0, 0, 0]

        def conditional_set(x):
            if torch.is_inference_mode_enabled():
                order[x] = 1  # no grad
            else:
                order[x] = 2  # with grad

        def create(layers, mode):
            return ExtendableSheafGCNLayer(5, 5,
                                           layers,
                                           epochs_per_operator=6,
                                           operator_compute_train_mode=mode)

        first = LambdaModule(lambda x: conditional_set(x))
        second = LambdaModule(lambda x: conditional_set(x))
        third = LambdaModule(lambda x: conditional_set(x))

        sheaf_layer = create([first, second, third], OperatorComputeLayerTrainMode.CONSECUTIVE)

        sheaf_layer.compute_layer(0, first, x=0)
        sheaf_layer.compute_layer(1, second, x=1)
        sheaf_layer.compute_layer(2, third, x=2)

        assert order == [2, 0, 0]

        sheaf_layer = create([first, second, third], OperatorComputeLayerTrainMode.CONSECUTIVE)
        sheaf_layer.set_current_epoch(7)

        sheaf_layer.compute_layer(0, first, x=0)
        sheaf_layer.compute_layer(1, second, x=1)
        sheaf_layer.compute_layer(2, third, x=2)

        assert order == [1, 2, 0]

        sheaf_layer = create([first, second, third], OperatorComputeLayerTrainMode.INCREMENTAL)
        sheaf_layer.set_current_epoch(7)

        sheaf_layer.compute_layer(0, first, x=0)
        sheaf_layer.compute_layer(1, second, x=1)
        sheaf_layer.compute_layer(2, third, x=2)

        assert order == [2, 2, 0]

        sheaf_layer = create([first, second, third], OperatorComputeLayerTrainMode.SIMULTANEOUS)
        sheaf_layer.set_current_epoch(7)

        sheaf_layer.compute_layer(0, first, x=0)
        sheaf_layer.compute_layer(1, second, x=1)
        sheaf_layer.compute_layer(2, third, x=2)

        assert order == [2, 2, 2]
