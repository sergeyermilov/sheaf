import numpy as np
from torch import nn, optim
from torch.autograd import Variable

def compute_loss_weights_simple(loss_orth, loss_bpr, nbatch, kappa=1.0e0, eps=1.0e-6):
    w_orth = 1.0
    w_bpr = 1.0

    loss_orth_numpy = (0.0 + loss_orth).cpu().detach().numpy()
    loss_bpr_numpy = (0.0 + loss_bpr).cpu().detach().numpy()

    w_bpr = np.exp(-np.sqrt(nbatch) * kappa * loss_orth_numpy)
    w_summ = w_orth + w_bpr
    w_orth = w_orth / w_summ
    w_bpr = w_bpr / w_summ

    return (w_orth, w_bpr)


def compute_neural_network_parameters(nnet, nepoch, nbatch, lr, w, x):
    nnet.fc_smat_pair.requires_grad_(True)
    print('nbatch = ' + str(nbatch))
    optimizer = optim.Adam(nnet.parameters(), lr=lr, weight_decay=0.0)
    loss_data = np.zeros((nepoch + 1))
    for kepoch in range(nepoch):
        optimizer.zero_grad()
        loss_orth, loss_cons, loss_smap  = nnet(w, Variable(x))
        w_orth, w_cons, w_smap = compute_loss_weights_simple(loss_orth, loss_cons, loss_smap, nbatch)
        loss = w_orth * loss_orth + w_cons * loss_cons + w_smap * loss_smap
        loss_data[kepoch] = loss.item()
        loss.backward()
        print('kepoch = ' + str(kepoch) + ' w_orth = ' + str(w_orth))
        print('kepoch = ' + str(kepoch) + ' w_cons = ' + str(w_cons))
        print('kepoch = ' + str(kepoch) + ' w_smap = ' + str(w_smap))
        print('kepoch = ' + str(kepoch) + ' loss_orth.item() = ' + str(loss_orth.item()))
        print('kepoch = ' + str(kepoch) + ' loss_cons.item() = ' + str(loss_cons.item()))
        print('kepoch = ' + str(kepoch) + ' loss_smap.item() = ' + str(loss_smap.item()))
        print('kepoch = ' + str(kepoch) + ' loss.item() = ' + str(loss.item()))
        optimizer.step()

    loss_orth, loss_cons, loss_smap  = nnet(w, Variable(x))
    w_orth, w_cons, w_smap = compute_loss_weights_simple(loss_smap, loss_cons, loss_orth, nbatch)
    loss = w_orth * loss_orth + w_cons * loss_cons + w_smap * loss_smap
    loss_data[-1] = loss.item()
    return loss_data