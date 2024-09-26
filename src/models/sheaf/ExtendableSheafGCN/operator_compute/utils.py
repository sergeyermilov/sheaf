from torch import nn


def make_fc_transform(inpt: int, outpt: tuple[int, int], nsmat: int, depth: int = 6, dropout_proba: float = 0):
    assert len(outpt) == 2, "incorrect output dim"

    layers = [nn.Linear(inpt, nsmat), nn.ReLU()]

    for i in range(depth):
        layers.append(nn.Linear(nsmat, nsmat))
        if dropout_proba != 0:
            layers.append(nn.Dropout(dropout_proba))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(nsmat, outpt[0] * outpt[1]))
    return nn.Sequential(*layers)
