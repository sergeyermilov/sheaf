import numpy as np
import pytorch_lightning as pl

import scipy.sparse as sparse
from scipy.sparse import diags
from scipy.sparse.linalg import inv



class EASE(pl.LightningModule):
    def __init__(self,
                 lambda_reg,
                 dataset):
        super(EASE, self).__init__()
        df = dataset.pandas_data
        df["value"] = 1.0
        rows = df.user_id_idx
        cols = df.item_id_idx
        data = df.value

        coo = sparse.coo_matrix((data, (rows, cols)))
        print("EASE: Calculate grammian matrix")
        gramian_matrix = coo.transpose().dot(coo)
        reg_matrix = diags([lambda_reg] * gramian_matrix.shape[0], shape=gramian_matrix.shape)

        print("EASE: Begin invert matrix")
        inverse_matrix = inv(gramian_matrix + reg_matrix).toarray()
        print("EASE: Normalize weights")
        self.weights = inverse_matrix / (-np.diag(inverse_matrix))
        diag_indices = np.diag_indices(self.weights.shape[0])
        self.weights[diag_indices] = 0.0
        print("EASE: Training is successful")

    def training_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        pass

    def evaluate(self, interacted_user_id_idx, k):
        scores = np.sum(self.weights[interacted_user_id_idx], axis=0)
        scores[interacted_user_id_idx] = 0.0
        return scores.argsort()[::-1][:k]
