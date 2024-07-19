import numpy as np
from numba import jit
import torch

from nnet import Sheaf_NNet
from data_loader import generate_graph_weights_and_features
from sheaf_calculator import compute_neural_network_parameters

def learn_sheaf_parameters(dimy=3, numv=1000, nepoch=2048, lr=1.0e-4):
    print('learning_the_sheaf')
    w, x = generate_graph_weights_and_features(numv, dimy)
    dimx = 2 * dimy
    nnet = Sheaf_NNet(dimx, dimy).double()
    loss_data = compute_neural_network_parameters(nnet, nepoch, numv, lr, w, x)

    nnet_folder = './nnet_folder'
    fname = nnet_folder + '/nnet_sheaf.ptr'
    torch.save(nnet, fname)
    np.save('./results/loss_data.npy', loss_data)
    return 0


def performance_test():
    nnet_folder = './nnet_folder'
    fname = nnet_folder + '/nnet_sheaf.ptr'
    nnet = torch.load(fname)
    numv = 600
    dimy = 3
    w, x = generate_graph_weights_and_features(numv, dimy)
    smat = torch.reshape(nnet.fc_smat_pair(x), (-1, nnet.dimy, nnet.dimx))
    print(smat[0, :, :])
    s0 = smat[0, :, :].detach().cpu().numpy()
    print(np.dot(np.transpose(s0), s0))
    return 0

def main():
    print('inside the main function')
    learn_sheaf_parameters(dimy=3, numv=1200, nepoch=20000, lr=1.0e-3)
    performance_test()
    return 0

main()
