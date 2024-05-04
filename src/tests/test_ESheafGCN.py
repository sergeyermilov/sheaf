import torch
from unittest import TestCase

from torch_geometric.utils import to_torch_csc_tensor, to_dense_adj

from src.models.ESheafGCN import Sheaf_Conv_fixed


class TestSheaf_NNet(TestCase):
    def test_forward(self):
        x = torch.FloatTensor([[10.0, 20.0],
                               [20.0, 30.0],
                               [30.0, 40.0],
                               [40.0, 20.0]]
                              )

        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]])

        adj_matrix_sparse = torch.squeeze(to_dense_adj(edge_index))

        conv = Sheaf_Conv_fixed(2, 6, 6)
        a1, a2 = conv.forward(adj_matrix_sparse, x)
        print(a1, a2)

    def test(self):

        # Assuming you have the tensors 'smat' and 'y'
        smat = torch.randn(3, 4, 4)
        y = torch.randn(3, 4)

        result = torch.diagonal(torch.tensordot(smat, y, dims=([1], [1])), dim1=0, dim2=2)

        # Vectorizing the operation in one step
        vectorized_result = torch.einsum('ijk,ik->kj', smat, y)

        # Check if the results are the same
        print(result)
        print(vectorized_result)
