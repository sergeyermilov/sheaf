import torch
import pytorch_lightning as pl
from torch import nn

from src.losses.bpr import compute_bpr_loss


class SheafConvLayer(nn.Module):
    def __init__(self, num_nodes, edge_index, latent_dim=3):
        super(SheafConvLayer, self).__init__()
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.step_size = 1.0
        self.sheaf_learner = nn.Linear(2 * latent_dim, 1, bias=False)
        self.left_idx, self.right_idx = self.compute_left_right_map_index()
        self.linear = nn.Linear(latent_dim, latent_dim)

    def init_parameters(self):
        nn.init.normal_(self.linear.weight, std=0.1)
        nn.init.normal_(self.sheaf_learner.weight, std=0.1)

    def compute_left_right_map_index(self):
        """Computes indices for the full Laplacian matrix"""
        edge_to_idx = dict()
        for e in range(self.edge_index.size(1)):
            source = self.edge_index[0, e].item()
            target = self.edge_index[1, e].item()
            edge_to_idx[(source, target)] = e

        left_index, right_index = [], []
        row, col = [], []
        for e in range(self.edge_index.size(1)):
            source = self.edge_index[0, e].item()
            target = self.edge_index[1, e].item()
            left_index.append(e)
            right_index.append(edge_to_idx[(target, source)])

            row.append(source)
            col.append(target)

        left_index = torch.tensor(left_index, dtype=torch.long, device=self.edge_index.device)
        right_index = torch.tensor(right_index, dtype=torch.long, device=self.edge_index.device)
        left_right_index = torch.vstack([left_index, right_index])

        assert len(left_index) == self.edge_index.size(1)
        return left_right_index

    def build_laplacian(self, maps):
        row, col = self.edge_index

        left_maps = torch.index_select(maps, index=self.left_idx, dim=0)
        right_maps = torch.index_select(maps, index=self.right_idx, dim=0)
        non_diag_maps = -left_maps * right_maps

        # m_u = torch.zeros_like(embeddings)
        # indx = u.view(-1, 1).repeat(1, embeddings.shape[1])
        # m_u = torch.scatter_reduce(input=m_u, src=c_v, index=indx, dim=0, reduce="sum", include_self=False)
        diag_maps = torch.zeros_like(maps)
        torch.scatter_add(input=diag_maps, src=maps**2, index=row, dim=0, reduce="sum", include_self=False)
        #diag_maps = scatter_add(maps ** 2, row, dim=0, dim_size=self.num_nodes)

        d_sqrt_inv = (diag_maps + 1).pow(-0.5)
        left_norm, right_norm = d_sqrt_inv[row], d_sqrt_inv[col]
        norm_maps = left_norm * non_diag_maps * right_norm
        diag = d_sqrt_inv * diag_maps * d_sqrt_inv

        diag_indices = torch.arange(0, self.num_nodes, device=maps.device).view(1, -1).tile(2, 1)
        all_indices = torch.cat([diag_indices, self.edge_index], dim=-1)
        all_values = torch.cat([diag.view(-1), norm_maps.view(-1)])
        return torch.sparse_coo_tensor(all_indices, all_values, size=(self.num_nodes, self.num_nodes))

    def predict_restriction_maps(self, x):
        row, col = self.edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.sheaf_learner(torch.cat([x_row, x_col], dim=1))
        maps = torch.tanh(maps)
        return maps

    def forward(self, x):
        maps = self.predict_restriction_maps(x)
        laplacian = self.build_laplacian(maps)

        y = self.linear(x)
        x = x - self.step_size * torch.sparse.mm(laplacian, y)
        return x


class SheafGCN(pl.LightningModule):
    def __init__(self,
                 latent_dim,
                 dataset):
        super(SheafGCN, self).__init__()
        self.dataset = dataset
        self.embedding = nn.Embedding(dataset.num_users + dataset.num_items, latent_dim)
        num_nodes = dataset.num_items + dataset.num_users
        self.conv1 = SheafConvLayer(num_nodes, dataset.train_edge_index, latent_dim)
        self.conv2 = SheafConvLayer(num_nodes, dataset.train_edge_index, latent_dim)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.conv1.init_parameters()
        self.conv2.init_parameters()

    def forward(self, edge_index):
       emb0 = self.embedding.weight
       emb1 = self.conv1(emb0)

       embs = [emb0, emb1]
       out = (torch.mean(torch.stack(embs, dim=0), dim=0))
       return emb0, out

    def training_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        train_edge_index = self.dataset.train_edge_index
        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = self.encode_minibatch(users,
                                                                                          pos_items,
                                                                                          neg_items,
                                                                                          train_edge_index)
        bpr_loss, reg_loss = compute_bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)
        final_loss = bpr_loss + reg_loss
        self.log('final_loss', final_loss)
        self.log('bpr_loss', bpr_loss)
        self.log('reg_loss', reg_loss)
        return bpr_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, out = self.forward(edge_index)
        return (
           out[users],
           out[pos_items],
           out[neg_items],
           emb0[users],
           emb0[pos_items],
           emb0[neg_items])