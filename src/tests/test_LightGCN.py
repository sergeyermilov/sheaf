from unittest import TestCase

import torch

from src.models.LightGCN import LightGCNConv


class TestLightGCNConv(TestCase):
    def test_forward(self):
        # Create simple graph
        x = torch.FloatTensor([[1.0, 1.0],
                               [1.0, 1.0],
                               [1.0, 1.0],
                               [1.0, 1.0]])

        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]])
        conv = LightGCNConv()
        actual = conv.forward(x, edge_index)
        expected = torch.tensor([[0.7071, 0.7071],
                                 [1.2071, 1.2071],
                                 [1.2071, 1.2071],
                                 [0.7071, 0.7071]])
        torch.eq(actual, expected)

    def test_message(self):
        self.fail()
