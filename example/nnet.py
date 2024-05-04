import numpy as np
import torch
from torch import nn


class Sheaf_NNet(nn.Module):
    def __init__(self, dimx, dimy, nnet_list=[], nsmat=20):
        super(Sheaf_NNet, self).__init__()
        self.dimx = dimx
        self.dimy = dimy
        self.nnet_list = nnet_list

        self.fc_smat = nn.Sequential(nn.Linear(self.dimx, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, self.dimy * self.dimx))

    # def compute_projection(self, x):
    #     pmat = torch.reshape(self.fc_pmat(x), (-1, self.dimx - self.dimy, self.dimx))
    #     q = torch.transpose(torch.diagonal(torch.tensordot(pmat, x, dims=([2], [1])), dim1=0, dim2=2), 0, 1)
    #     return torch.transpose(torch.diagonal(torch.tensordot(pmat, q, dims=([1], [1])), dim1=0, dim2=2), 0, 1)

    def forward(self, w, x):
        smat = torch.reshape(self.fc_smat(x), (-1, self.dimy, self.dimx))
        print(w.shape)
        q = torch.transpose(torch.diagonal(torch.tensordot(smat, x, dims=([2], [1])), dim1=0, dim2=2), 0, 1)
        y = torch.tensordot(w, q, dims=([1], [0]))
        xmap = torch.transpose(torch.diagonal(torch.tensordot(smat, y, dims=([1], [1])), dim1=0, dim2=2), 0, 1)
        loss_smap = torch.mean((xmap - x) * (xmap - x)) * self.dimx

        rmat = torch.diagonal(torch.tensordot(smat, smat, dims=([2], [2])), dim1=0, dim2=2)
        target_matrix = torch.zeros((self.dimy, self.dimy, rmat.shape[-1]))
        for idx in range(target_matrix.shape[-1]):
            target_matrix[:, :, idx] = torch.eye(self.dimy)

        rmat = rmat - target_matrix
        loss_orth = torch.sqrt(torch.mean(rmat * rmat) * self.dimy * self.dimy)

        smat_proj = torch.reshape(self.fc_smat(xmap), (-1, self.dimy, self.dimx))
        loss_cons = torch.mean((smat_proj - smat) * (smat_proj - smat)) * self.dimy * self.dimx
        return (loss_orth, loss_cons, loss_smap)