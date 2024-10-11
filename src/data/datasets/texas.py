import torch
from dgl.data import TexasDataset

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import to_undirected

TEXAS_DATASET_RELATIVE_PATH = ""

class TexasTorchDataset(Dataset):
    def __init__(self, node_features, node_labels, train_edge_index, mask, random_state=42):
        self.node_features = node_features
        self.node_labels = node_labels
        self.num_classes = torch.unique(node_labels).shape[0]
        self.num_features = node_features.shape[1]
        self.train_edge_index = train_edge_index
        self.mask = mask
        self.mask_indexes = [i for i, x in enumerate(self.mask) if x]
        self.mask_indexes_dict = {k: v for k, v in enumerate(self.mask_indexes)}

    def __len__(self):
        return len(self.mask_indexes)

    def __getitem__(self, idx):
        return self.mask_indexes_dict[idx], self.node_features[self.mask_indexes_dict[idx]]


class TexasDataModule(LightningDataModule):
    def __init__(self,
                 dataset_path: str,
                 batch_size=32,
                 random_state=42,
                 device="cpu"):
        super().__init__()

        self.batch_size = batch_size
        dataset = TexasDataset()
        g = dataset[0]
        self.node_features = g.ndata["feat"]
        self.node_labels = g.ndata["label"]
        self.train_edge_index = to_undirected(torch.stack(g.edges("uv")))
        self.train_mask = g.ndata["train_mask"]
        self.val_mask = g.ndata["val_mask"]
        self.test_mask = g.ndata["test_mask"]

    def setup(self):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = TexasTorchDataset(self.node_features,
                                               self.node_labels,
                                               self.train_edge_index,
                                               self.train_mask[:, 0])

        self.val_dataset = TexasTorchDataset(self.node_features,
                                             self.node_labels,
                                             self.train_edge_index,
                                             self.val_mask[:, 0])

        self.test_dataset = TexasTorchDataset(self.node_features,
                                              self.node_labels,
                                              self.train_edge_index,
                                              self.test_mask[:, 0])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
