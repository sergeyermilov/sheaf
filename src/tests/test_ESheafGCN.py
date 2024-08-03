import torch
from unittest import TestCase

from torch_geometric.utils import to_torch_csc_tensor, to_dense_adj

from src.models.sheaf.ESheafGCN import ESheafLayer


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

        conv = ESheafLayer(2, 6, 6)
        product = conv.forward(adj_matrix_sparse, x)
        print(product)

    def test_2(self):

        # Assuming you have the tensors 'smat' and 'y'
        smat = torch.randn(4, 6, 2)

        result = torch.diagonal(torch.tensordot(smat, smat, dims=([2], [2])), dim1=0, dim2=2)
        print(result.shape)

        # Vectorizing the operation in one step
        vectorized_result = torch.einsum('ijk,ilk->lji', smat, smat)

        # Check if the results are the same
        print(result)
        print(vectorized_result)
