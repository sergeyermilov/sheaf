import torch
import numpy as np
from dgl.data import CoraFullDataset

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import to_undirected

CORA_FULL_DATASET_RELATIVE_PATH = ""
class CoraFullTorchDataset(Dataset):
    def __init__(self, node_features, node_labels, train_edge_index, mask, random_state=42):
        self.node_features = node_features
        self.node_labels = node_labels
        self.num_classes = 70
        self.num_features = node_features.shape[1]
        self.train_edge_index = train_edge_index
        self.mask = mask
        self.mask_indexes = [i for i, x in enumerate(self.mask) if x]
        self.mask_indexes_dict = {k: v for k, v in enumerate(self.mask_indexes)}

    def __len__(self):
        return len(self.mask_indexes)

    def __getitem__(self, idx):
        return self.mask_indexes_dict[idx], self.node_features[self.mask_indexes_dict[idx]]


class CoraFullDataModule(LightningDataModule):
    def __init__(self,
                 dataset_path: str,
                 batch_size=32,
                 random_state=42,
                 device="cpu"):
        super().__init__()

        self.batch_size = batch_size
        dataset = CoraFullDataset()
        g = dataset[0]
        self.node_features = g.ndata["feat"]
        self.node_labels = g.ndata["label"]
        self.train_edge_index = to_undirected(torch.stack(g.edges("uv")))

        self.train_mask, self.val_mask, self.test_mask = self.generate_train_test_split_mask( self.node_features.shape[0])

    def generate_train_test_split_mask(self, n_samples):
        # Generate an array of indices
        indices = np.arange(n_samples)

        # Shuffle the indices
        np.random.shuffle(indices)

        # Calculate the split indices
        train_split_index = int(n_samples * 0.6)
        val_split_index = int(n_samples * 0.8)

        # Generate the masks
        train_mask = np.zeros(n_samples, dtype=bool)
        val_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)

        train_mask[:train_split_index] = True
        val_mask[train_split_index:val_split_index] = True
        test_mask[val_split_index:] = True

        # Shuffle the masks
        np.random.shuffle(train_mask)
        np.random.shuffle(val_mask)
        np.random.shuffle(test_mask)
        return train_mask, val_mask, test_mask

    def setup(self):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = CoraFullTorchDataset(self.node_features,
                                                  self.node_labels,
                                                  self.train_edge_index,
                                                  self.train_mask)

        self.val_dataset = CoraFullTorchDataset(self.node_features,
                                                self.node_labels,
                                                self.train_edge_index,
                                                self.val_mask)

        self.test_dataset = CoraFullTorchDataset(self.node_features,
                                                 self.node_labels,
                                                 self.train_edge_index,
                                                 self.test_mask)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
