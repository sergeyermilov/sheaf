import torch
from torch import nn

from src.models.sheaf.ExtendableSheafGCN.operator_compute import OperatorComputeLayer, SheafOperators


class OperatorComputeLayerTrainMode:
    CONSECUTIVE = "cons"  # second, but not first, and not third yet
    INCREMENTAL = "inc"  # first and second, but not third
    SIMULTANEOUS = "sim"  # first and second and third


class ExtendableSheafGCNLayer(nn.Module):
    def __init__(self,
                 dimx: int,
                 dimy: int,
                 operator_compute_layers: list[OperatorComputeLayer],
                 operator_compute_train_mode: str = OperatorComputeLayerTrainMode.SIMULTANEOUS,
                 epochs_per_operator: int = None):
        super(ExtendableSheafGCNLayer, self).__init__()
        self.dimx = dimx
        self.dimy = dimy
        self.operator_compute_layers = nn.ModuleList(operator_compute_layers)
        self.epochs_per_operator = epochs_per_operator
        self.train_mode = operator_compute_train_mode

        self.orth_eye = torch.eye(self.dimy).unsqueeze(0)
        self.current_epoch = 0

    @staticmethod
    def compute_sheaf(A_uv_t, A_v_u, embeddings, indices) -> torch.Tensor:
        #########################################
        ## compute h_v = A(u,v)^T * A(v,u) * x(v)
        #########################################
        x_v = embeddings[indices, ...]
        # compute A(v,u) * x(v)
        h_v_ = torch.bmm(A_v_u, x_v.unsqueeze(-1))
        # compute h_v = A(u,v)^T * A(v,u) * x(v)
        h_v = torch.bmm(A_uv_t, h_v_).squeeze(-1)
        #########################################

        return h_v

    @staticmethod
    def scale_sheaf(adj_matrix, u_indices, v_indices, h_v) -> torch.Tensor:
        ############################
        # compute c_v = w(v,u) * h_v
        ############################
        # extract w(v, u)
        embedding_weights = adj_matrix[v_indices, u_indices]
        # c_v = w(v, u) * h_v
        c_v = embedding_weights.view(-1, 1) * h_v
        #########################################

        return c_v

    @staticmethod
    def compute_message(embeddings, u_indices, sheafs):
        ############################
        # compute  sum_v
        ############################
        m_u = torch.zeros_like(embeddings)
        indx = u_indices.view(-1, 1).repeat(1, embeddings.shape[1])
        # sum c_v for each u
        return torch.scatter_reduce(
            input=m_u,
            src=sheafs,
            index=indx,
            dim=0,
            reduce="sum",
            include_self=False
        )

    @staticmethod
    def compute_diff_loss(messages, embeddings):
        diff_x = (messages - embeddings)
        diff_x_t = diff_x.swapaxes(-1, -2)
        diff_w = torch.mm(diff_x_t, diff_x)

        return diff_w.mean()

    @staticmethod
    def compute_cons_loss(embeddings, u_indices, A_uv, A_uv_t):
        embeddings_u = embeddings[u_indices, ...]
        x = embeddings_u.unsqueeze(-1)
        x_t = embeddings_u.unsqueeze(-1).swapaxes(-1, -2)

        # P(u, v) = A(u, v)^T A(u, v)
        cons_p = torch.bmm(A_uv_t, A_uv)
        # A(u, v) - A(u, v) P(u, v)
        cons_y = A_uv - torch.bmm(A_uv, cons_p)
        # Q(u, v) = (A(u, v) - A(u, v) P(u, v))^T (A(u, v) - A(u, v) P(u, v))
        cons_q = torch.bmm(cons_y.swapaxes(-1, -2), cons_y)
        # W(u, v) = x(u)^T Q(u, v) x(u)
        cons_w1 = torch.bmm(cons_q, x)
        cons_w2 = torch.bmm(x_t, cons_w1)

        return cons_w2.mean()

    @staticmethod
    def compute_orth_loss(A_uv, A_uv_t, orth_eye):
        # compute intermediate values for loss orth
        orth_aat = torch.bmm(A_uv, A_uv_t)
        orth_q = orth_aat - orth_eye
        orth_z = torch.bmm(orth_q.swapaxes(-1, -2), orth_q)

        # compute trace
        orth = torch.einsum("ijj", orth_z)

        return torch.mean(orth)

    def compute_layer(self, layer_ix, layer, **params):
        expected_layer_ix = self.current_epoch % self.epochs_per_operator

        def infer_no_grad():
            with torch.inference_mode():
                return layer(**params)

        def infer():
            return layer(**params)

        match self.train_mode:
            case OperatorComputeLayerTrainMode.SIMULTANEOUS:
                return infer()
            case OperatorComputeLayerTrainMode.INCREMENTAL:
                if layer_ix <= expected_layer_ix:
                    return infer()
            case OperatorComputeLayerTrainMode.CONSECUTIVE:
                if layer_ix < expected_layer_ix:
                    return infer_no_grad()

                if layer_ix == expected_layer_ix:
                    return infer()

        return None

    # Use \hat{x} = A^T * A * x instead of message passing,
    # only available for no-/single- feature computers
    def denoise(self, embeddings: torch.Tensor) -> torch.Tensor:
        operators = torch.zeros((embeddings.shape[0], self.dimy, self.dimx), requires_grad=False)
        for layer_ix, operator_compute_layer in enumerate(sorted(self.operator_compute_layers, key=lambda x: x.priority())):
            with torch.inference_mode():
                if not operator_compute_layer.is_denoisable():
                    raise Exception(f"operator {operator_compute_layer} is not denoisable.")
                operators = operator_compute_layer.compute_for_denoise(embeddings, operators)

        A = operators
        A_t = A.swapaxes(-1, -2)  # A(u, v)^T
        D = torch.bmm(A_t, A) # denoise operator
        denoised_embeddings = torch.bmm(D, embeddings.unsqueeze(-1))
        # remove extra redundant dimension
        return denoised_embeddings.squeeze(-1)

    def is_denoisable(self):
        return all(layer.is_denoisable() for layer in self.operator_compute_layers)

    def forward(self, adj_matrix: torch.Tensor, embeddings: torch.Tensor, edge_index: torch.Tensor, compute_losses: bool = False):
        u_indices = edge_index[0, :]
        v_indices = edge_index[1, :]

        sheaf_operators = SheafOperators(
            torch.zeros((edge_index.shape[1], self.dimy, self.dimx), requires_grad=False),
            torch.zeros((edge_index.shape[1], self.dimy, self.dimx), requires_grad=False)
        )

        for layer_ix, operator_compute_layer in enumerate(sorted(self.operator_compute_layers, key=lambda x: x.priority())):
            sheaf_operators_updated = self.compute_layer(
                layer_ix, operator_compute_layer,
                operators=sheaf_operators, embeddings=embeddings, u_indices=u_indices, v_indices=v_indices
            )

            if sheaf_operators_updated is not None:
                sheaf_operators = sheaf_operators_updated

        A_uv = sheaf_operators.operator_uv
        A_vu = sheaf_operators.operator_vu

        A_uv_t = torch.reshape(A_uv, (-1, self.dimy, self.dimx)).swapaxes(-1, -2)  # A(u, v)^T

        h_v = ExtendableSheafGCNLayer.compute_sheaf(A_uv_t=A_uv_t, A_v_u=A_vu, embeddings=embeddings, indices=v_indices)
        c_v = ExtendableSheafGCNLayer.scale_sheaf(adj_matrix=adj_matrix, u_indices=u_indices, v_indices=v_indices, h_v=h_v)
        m_u = ExtendableSheafGCNLayer.compute_message(embeddings=embeddings, u_indices=u_indices, sheafs=c_v)

        if not compute_losses:
            return m_u

        diff_loss = ExtendableSheafGCNLayer.compute_diff_loss(messages=m_u, embeddings=embeddings)
        cons_loss = ExtendableSheafGCNLayer.compute_cons_loss(embeddings=embeddings, u_indices=u_indices, A_uv=A_uv, A_uv_t=A_uv_t)
        orth_loss = ExtendableSheafGCNLayer.compute_orth_loss(A_uv=A_uv, A_uv_t=A_uv_t, orth_eye=self.orth_eye)

        return m_u, diff_loss, cons_loss, orth_loss

    def init_parameters(self):
        for layer in self.operator_compute_layers:
            layer.init_parameters()

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch
