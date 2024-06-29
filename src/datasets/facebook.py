import torch
import random
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing as pp
from torch_geometric.utils import to_dense_adj


class FacebookDataset(Dataset):
    def __init__(self, df):
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

        self.adjacency_matrix = torch.squeeze(to_dense_adj(self.train_edge_index, max_num_nodes=self.num_items + self.num_users))

    def __len__(self):
        return self.pandas_data.shape[0]

    def __getitem__(self, idx):
        user_idx = self.pandas_data["user_id_idx"].iloc[idx]
        row = self.interacted_items_by_user_idx.iloc[user_idx]
        user_idx = row["user_id_idx"]
        pos_item_idx = random.choice(row["item_id_idx"])
        neg_item_idx = self.sample_neg(row["item_id_idx"])
        return torch.tensor(user_idx), torch.tensor(pos_item_idx + self.num_users), torch.tensor(neg_item_idx + self.num_users)

    def sample_neg(self, x):
        while True:
            neg_id = random.randint(0, self.num_items - 1)
            if neg_id not in x:
                return neg_id

class FacebookDataModule(LightningDataModule):
    def __init__(self, ratings_file, sep='\t', batch_size=32):
        super().__init__()
        self.batch_size = batch_size

        COLUMNS_NAME = ['user_id', 'item_id', 'rating']
        self.pandas_data = pd.read_csv(ratings_file, sep=sep, names=COLUMNS_NAME, engine='python')

        # Train/val/test splitting
        train, test = train_test_split(self.pandas_data, test_size=0.2)
        val, test = train_test_split(test, test_size=0.5)
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
        self.train_dataset = FacebookDataset(self.train_df)
        self.val_dataset = FacebookDataset(self.val_df)
        self.test_dataset = FacebookDataset(self.test_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)