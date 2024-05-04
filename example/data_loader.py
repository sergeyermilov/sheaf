import numpy as np
import torch

def generate_graph_weights_and_features(numv, dimy, gamma_ref=1.0):
    ### numv number of verticies
    ### dimy -dimension of the ambient space
    dimx = 2 * dimy
    x = np.random.normal(0.0, 1.0, (numv, dimx))
    x[:, :dimy] = 0.1 + np.random.rand(numv, dimy)
    for idx in range(numv):
        x[:, :dimy] = x[:, :dimy] / np.sqrt(np.sum(x[:, :dimy] * x[:, :dimy]))
    gamma = gamma_ref / (numv ** (1.0 / dimy))
    w = np.zeros((numv, numv))
    for idx0 in range(numv):
        for idx1 in range(numv):
            dy = x[idx1, :dimy] - x[idx0, :dimy]
            w[idx0, idx1] = np.exp(- gamma * np.sum(dy * dy))
        w[idx0, :] = w[idx0, :] / np.sum(w[idx0, :])
    return (torch.from_numpy(w).double(), torch.from_numpy(x).double())


print(generate_graph_weights_and_features(10, 3))