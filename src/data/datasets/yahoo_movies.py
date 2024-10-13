import torch
import random
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing as pp

from src.models.graph_utils import k_hop_subgraph_limit

YAHOO_DATASET_RELATIVE_PATH = "yahoo/ydata-ymovies-user-movie-ratings-train-v1_0.txt"


class YahooMoviesDataset(Dataset):
    def __init__(self,
                 df,
                 num_users,
                 num_items,
                 random_state=42,
                 enable_subsampling=False,
                 num_k_hops=2,
                 hop_max_edges=1000,
                 device='cpu'):
        random.seed(random_state)
        self.enable_subsampling = enable_subsampling
        self.num_k_hops = num_k_hops
        self.hop_max_edges = hop_max_edges
        self.device = device
        self.random_state = random_state

        # Unique user and items ids as numpy array
        self.pandas_data = df

        self.num_users = num_users
        self.num_items = num_items

        # Create graph
        self.interacted_items_by_user_idx = self.pandas_data.groupby('user_id_idx')['item_id_idx'].apply(list)

        u_t = torch.tensor(self.pandas_data.user_id_idx.values, dtype=torch.long)
        i_t = torch.tensor(self.pandas_data.item_id_idx.values, dtype=torch.long) + self.num_users

        self.train_edge_index = torch.stack((
            torch.cat([u_t, i_t]),
            torch.cat([i_t, u_t])
        ))

        # self.adjacency_matrix = torch.squeeze(to_dense_adj(self.train_edge_index, max_num_nodes=self.num_items + self.num_users))

    def __len__(self):
        return self.pandas_data.shape[0]

    def __getitem__(self, idx):
        user_idx = self.pandas_data["user_id_idx"].iloc[idx]
        row = self.interacted_items_by_user_idx.loc[user_idx]
        pos_item_idx = random.choice(row)
        neg_item_idx = self.sample_neg(row)
        return (torch.tensor(user_idx),
                torch.tensor(pos_item_idx + self.num_users),
                torch.tensor(neg_item_idx + self.num_users))

    def k_hop_subgraph(self, user_idxs_tensor, num_hops):
        sub_edge_index, _ = k_hop_subgraph_limit(
            node_idx=user_idxs_tensor,
            num_hops=num_hops,
            edge_index=self.train_edge_index,
            hop_max_edges=self.hop_max_edges,
            rng=torch.Generator(device=self.device).manual_seed(self.random_state)
        )

        return sub_edge_index

    def __getitems__(self, indices):
        sample = self.pandas_data.iloc[indices]
        user_idxs = sample["user_id_idx"].values
        user_idxs_tensor = torch.tensor(user_idxs)
        sample_interacted_items = self.interacted_items_by_user_idx.loc[user_idxs]
        pos_item_idxs = sample_interacted_items.apply(lambda x: random.choice(x)).values
        neg_item_idxs = sample_interacted_items.apply(lambda x: random.randint(0, self.num_items - 1)).values

        if self.enable_subsampling:
            sub_edge_index = self.k_hop_subgraph(user_idxs_tensor, self.num_k_hops)
        else:
            sub_edge_index = self.train_edge_index

        return torch.tensor(user_idxs), torch.tensor(pos_item_idxs), torch.tensor(neg_item_idxs), sub_edge_index

    def get_num_nodes(self):
        return self.num_users + self.num_items


class YahooMoviesDataModule(LightningDataModule):
    def __init__(self,
                 dataset_path: str,
                 sep='\t',
                 batch_size=32,
                 random_state=42,
                 split="simple",
                 device="cpu",
                 enable_subsampling=False,
                 num_k_hops=2,
                 hop_max_edges=1000,
                 num_workers=6,
                 ):
        super().__init__()
        if split != "simple":
            raise NotImplementedError("Only simple split is available.")

        self.batch_size = batch_size
        self.random_state = random_state
        self.enable_subsampling = enable_subsampling
        self.num_k_hops = num_k_hops
        self.hop_max_edges = hop_max_edges
        self.device = device
        self.num_workers = num_workers

        COLUMNS_NAME = ['user_id', 'item_id', 'full_rating', "rating"]
        self.pandas_data = pd.read_csv(dataset_path, sep=sep, names=COLUMNS_NAME, engine='python')
        self.pandas_data = self.pandas_data[self.pandas_data['rating'] >= 3]

        self.num_users = len(self.pandas_data.user_id.unique())
        self.num_items = len(self.pandas_data.item_id.unique())

        # Train/val/test splitting
        train, test = train_test_split(self.pandas_data, test_size=0.2, random_state=random_state)
        val, test = train_test_split(test, test_size=0.5, random_state=random_state)
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
        self.train_dataset = YahooMoviesDataset(self.train_df,
                                                num_users=self.num_users,
                                                num_items=self.num_items,
                                                enable_subsampling=self.enable_subsampling,
                                                num_k_hops=self.num_k_hops,
                                                hop_max_edges=self.hop_max_edges,
                                                device=self.device)
        self.val_dataset = YahooMoviesDataset(self.val_df,
                                              num_users=self.num_users,
                                              num_items=self.num_items,
                                              enable_subsampling=self.enable_subsampling,
                                              num_k_hops=self.num_k_hops,
                                              hop_max_edges=self.hop_max_edges,
                                              device=self.device)
        self.test_dataset = YahooMoviesDataset(self.test_df,
                                               num_users=self.num_users,
                                               num_items=self.num_items,
                                               enable_subsampling=self.enable_subsampling,
                                               num_k_hops=self.num_k_hops,
                                               hop_max_edges=self.hop_max_edges,
                                               device=self.device)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          pin_memory=False,
                          shuffle=True,
                          generator=torch.Generator(device=self.device).manual_seed(self.random_state),
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          pin_memory=False,
                          shuffle=False,
                          generator=torch.Generator(device=self.device).manual_seed(self.random_state),
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          pin_memory=False,
                          shuffle=False,
                          generator=torch.Generator(device=self.device).manual_seed(self.random_state),
                          num_workers=self.num_workers)

    def collate_fn(self, batch):
        return batch
