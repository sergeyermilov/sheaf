import numpy as np
import torch


def compute_loss_weight_paper(loss_diff: torch.Tensor,
                              loss_orth: torch.Tensor,
                              loss_cons: torch.Tensor,
                              nbatch: int,
                              kappa: float = 0.01e0):

    with torch.no_grad():
        w_orth = 1.0

        loss_diff = loss_diff.detach().clone()
        loss_orth = loss_orth.detach().clone()
        loss_cons = loss_cons.detach().clone()

        nbatch_sqrt = torch.sqrt(torch.tensor(nbatch))

        w_cons = torch.exp(-kappa * nbatch_sqrt * loss_orth)
        w_diff = torch.exp(-kappa * nbatch_sqrt * torch.max(torch.tensor([loss_orth, loss_cons])))
        w_bpr = torch.exp(-kappa * nbatch_sqrt * torch.max(torch.tensor([loss_orth, loss_cons, loss_diff])))

        w_summ = w_orth + w_diff + w_cons + w_bpr

        w_orth = w_orth / w_summ
        w_diff = w_diff / w_summ
        w_cons = w_cons / w_summ
        w_bpr = w_bpr / w_summ

        return w_diff, w_orth, w_cons, w_bpr


def compute_loss_weights_simple(loss_smap, loss_orth, loss_cons, loss_bpr, nbatch, kappa=0.01e0, eps=1.0e-6):
    w_smap = 1.0
    w_orth = 1.0
    w_bpr = 1.0
    w_cons = 1.0


    loss_smap_numpy = (0.0 + loss_smap).cpu().detach().numpy()
    loss_orth_numpy = (0.0 + loss_orth).cpu().detach().numpy()
    loss_cons_numpy = (0.0 + loss_cons).cpu().detach().numpy()
    loss_bpr_numpy = (0.0 + loss_bpr).cpu().detach().numpy()


    w_orth = np.exp(-np.sqrt(nbatch) * kappa * loss_smap_numpy)
    w_cons = np.exp(-np.sqrt(nbatch) * kappa * max(loss_smap_numpy, loss_orth_numpy))
    w_bpr = np.exp(-np.sqrt(nbatch) * kappa * max(loss_smap_numpy, loss_orth_numpy, loss_cons_numpy))

    w_summ = w_smap + w_orth + w_cons + w_bpr
    w_smap = w_smap/w_summ
    w_orth = w_orth / w_summ
    w_cons = w_cons / w_summ
    w_bpr = w_bpr / w_summ

    return (w_smap, w_orth, w_cons, w_bpr)
