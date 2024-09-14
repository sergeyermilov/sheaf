import torch
import random
import pathlib
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing as pp

from src.data.utils import extract_from_archive
from src.models.graph_utils import k_hop_subgraph_limit

RATINGS_FILE_CSV = "ratings.csv"

MOVIE_LENS_1M_DATASET_RELATIVE_PATH = "ml-1m/ml-1m.tar.gz"
MOVIE_LENS_10M_DATASET_RELATIVE_PATH = "ml-10m/ml-10m.tar.gz"

# Move to external params
NUM_K_HOPS = 2


class MovieLensDataset(Dataset):
    def __init__(self, df,
                 random_state=42,
                 enable_subsampling=False,
                 num_k_hops=2,
                 hop_max_edges=1000):
        random.seed(random_state)
        self.enable_subsampling = enable_subsampling
        self.num_k_hops = num_k_hops
        self.hop_max_edges = hop_max_edges

        # Unique user and items ids as numpy array
        self.pandas_data = df
        self.user_ids = self.pandas_data.user_id.unique()
        self.item_ids = self.pandas_data.item_id.unique()

        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)

        # Create graph
        self.interacted_items_by_user_idx = self.pandas_data.groupby('user_id_idx')['item_id_idx'].apply(
            list).reset_index()

        u_t = torch.tensor(self.pandas_data.user_id_idx.values, dtype=torch.long)
        i_t = torch.tensor(self.pandas_data.item_id_idx.values, dtype=torch.long) + self.num_users

        self.train_edge_index = torch.stack((
            torch.cat([u_t, i_t]),
            torch.cat([i_t, u_t])
        ))

        # self.adjacency_matrix = torch.squeeze(to_dense_adj(self.train_edge_index, max_num_nodes=self.num_items + self.num_users))
        # self.adjacency_map = convert_edge_index_to_adjacency_map(self.train_edge_index)

    def __len__(self):
        return self.pandas_data.shape[0]

    def __getitem__(self, idx):
        user_idx = self.pandas_data["user_id_idx"].iloc[idx]
        row = self.interacted_items_by_user_idx.iloc[user_idx]
        user_idx = row["user_id_idx"]
        pos_item_idx = random.choice(row["item_id_idx"])
        neg_item_idx = self.sample_neg(row["item_id_idx"])
        return torch.tensor(user_idx), torch.tensor(pos_item_idx + self.num_users), torch.tensor(
            neg_item_idx + self.num_users)

    def k_hop_subgraph(self, user_idxs_tensor, num_hops, train_edge_index, relabel_nodes=False):
        _, sub_edge_index, _, _ = k_hop_subgraph_limit(user_idxs_tensor, num_hops, self.train_edge_index, hop_max_edges = self.hop_max_edges)
        return sub_edge_index

    def __getitems__(self, indices):
        sample = self.pandas_data.iloc[indices]
        user_idxs = sample["user_id_idx"].values
        user_idxs_tensor = torch.tensor(user_idxs)
        sample_interacted_items = self.interacted_items_by_user_idx.iloc[user_idxs]
        pos_item_idxs = sample_interacted_items["item_id_idx"].apply(lambda x: random.choice(x)).values
        neg_item_idxs = sample_interacted_items["item_id_idx"].apply(lambda x: self.sample_neg(x)).values

        if self.enable_subsampling:
            sub_edge_index = self.k_hop_subgraph(user_idxs_tensor, NUM_K_HOPS, self.train_edge_index)
        else:
            sub_edge_index = self.train_edge_index

        return torch.tensor(user_idxs), torch.tensor(pos_item_idxs), torch.tensor(neg_item_idxs), sub_edge_index

    def sample_neg(self, x):
        while True:
            neg_id = random.randint(0, self.num_items - 1)
            if neg_id not in x:
                return neg_id


class MovieLensDataModule(LightningDataModule):
    def __init__(self,
                 dataset_path: str,
                 sep='::',
                 device="cpu",
                 batch_size=32,
                 random_state=42,
                 split="simple",
                 enable_subsampling=False,
                 num_k_hops=2,
                 hop_max_edges=1000
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.random_state = random_state
        self.enable_subsampling = enable_subsampling
        self.num_k_hops = num_k_hops
        self.hop_max_edges = hop_max_edges
        self.device = device

        dataset_path = pathlib.Path(dataset_path)
        extract_from_archive(dataset_path, [RATINGS_FILE_CSV], dataset_path.parent)

        # we have only one file
        ratings_file = dataset_path.parent / RATINGS_FILE_CSV

        COLUMNS_NAME = ['user_id', 'item_id', 'rating', "timestamp"]
        self.pandas_data = pd.read_csv(ratings_file, sep=sep, names=COLUMNS_NAME, engine='python')
        self.pandas_data = self.pandas_data[self.pandas_data['rating'] >= 3]

        # Train/val/test splitting
        if split == "simple":
            train, test = train_test_split(self.pandas_data, test_size=0.2, random_state=random_state)
            val, test = train_test_split(test, test_size=0.5, random_state=random_state)
        elif split == "time":
            sorted_data = self.pandas_data.sort_values(by="timestamp", ascending=False)
            train_thresh = int(sorted_data.shape[0] * 0.8)
            test_thresh = train_thresh + int(sorted_data.shape[0] * 0.1)
            train = sorted_data.iloc[:train_thresh]
            test = sorted_data.iloc[train_thresh:test_thresh]
            val = sorted_data.iloc[test_thresh:]

        self.train_df = pd.DataFrame(train, columns=self.pandas_data.columns)
        self.val_df = pd.DataFrame(val, columns=self.pandas_data.columns)
        self.test_df = pd.DataFrame(test, columns=self.pandas_data.columns)

        label_encoder_user = pp.LabelEncoder()
        label_encoder_item = pp.LabelEncoder()
        self.train_df["user_id_idx"] = label_encoder_user.fit_transform(self.train_df['user_id'].values)
        self.train_df["item_id_idx"] = label_encoder_item.fit_transform(self.train_df['item_id'].values)

        train_user_ids = self.train_df['user_id'].unique()
        train_item_ids = self.train_df['item_id'].unique()

        self.val_df = self.val_df[
            (self.val_df['user_id'].isin(train_user_ids)) & \
            (self.val_df['item_id'].isin(train_item_ids))
            ]

        self.val_df['user_id_idx'] = label_encoder_user.transform(self.val_df['user_id'].values)
        self.val_df['item_id_idx'] = label_encoder_item.transform(self.val_df['item_id'].values)

        self.test_df = self.test_df[
            (self.test_df['user_id'].isin(train_user_ids)) & \
            (self.test_df['item_id'].isin(train_item_ids))
            ]

        self.test_df['user_id_idx'] = label_encoder_user.transform(self.test_df['user_id'].values)
        self.test_df['item_id_idx'] = label_encoder_item.transform(self.test_df['item_id'].values)

    def setup(self):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = MovieLensDataset(self.train_df,
                                              enable_subsampling=self.enable_subsampling,
                                              num_k_hops=self.num_k_hops,
                                              hop_max_edges=self.hop_max_edges
                                              )
        self.val_dataset = MovieLensDataset(self.val_df,
                                            enable_subsampling=self.enable_subsampling,
                                            num_k_hops=self.num_k_hops,
                                            hop_max_edges=self.hop_max_edges)

        self.test_dataset = MovieLensDataset(self.test_df,
                                             enable_subsampling=self.enable_subsampling,
                                             num_k_hops=self.num_k_hops,
                                             hop_max_edges=self.hop_max_edges)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          pin_memory=False,
                          shuffle=True,
                          generator=torch.Generator(device=self.device)
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          pin_memory=False,
                          shuffle=True,
                          generator=torch.Generator(device=self.device)
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          pin_memory=False,
                          shuffle=True,
                          generator=torch.Generator(device=self.device)
                          )

    def collate_fn(self, batch):
        return batch
