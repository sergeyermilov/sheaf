from unittest import TestCase

import torch

from src.models.sheaf.SheafGCN import SheafConvLayer


class TestSheafConvLayer(TestCase):

    def test_compute_left_right_map_index(self):
        x = torch.FloatTensor([[1.0, 1.0],
                               [1.0, 1.0],
                               [1.0, 1.0],
                               [1.0, 1.0]])

        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]])
        conv = SheafConvLayer(4)
        print(conv.compute_left_right_map_index(edge_index))

    def test_predict_restriction_maps(self):
        x = torch.FloatTensor([[1.0, 0.5, 0.0],
                               [1.0, 0.0, 0.5],
                               [0.5, 0.0, 1.0],
                               [1.0, 1.0, 0.5]])

        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]])

        conv = SheafConvLayer(4)
        maps = conv.predict_restriction_maps(x, edge_index)
        print(conv.build_laplacian(x, maps, edge_index))
