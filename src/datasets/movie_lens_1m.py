import torch
import random
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing as pp
from torch_geometric.utils import to_dense_adj
import numpy as np
import torch.nn as nn
import torch

MOVIE_LENS_DATASET_RELATIVE_PATH = "ml-1m/ratings.dat"


class MovieLensDataset(Dataset):
    def __init__(self, df, random_state=42):
        random.seed(random_state)

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

class MovieLensDataModule(LightningDataModule):
    def __init__(self, ratings_file, sep='::', batch_size=32, random_state=42):
        super().__init__()
        self.batch_size = batch_size

        COLUMNS_NAME = ['user_id', 'item_id', 'rating', "timestamp"]
        self.pandas_data = pd.read_csv(ratings_file, sep=sep, names=COLUMNS_NAME, engine='python')
        self.pandas_data = self.pandas_data[self.pandas_data['rating'] >= 3]

        # Train/val/test splitting
        train, test = train_test_split(self.pandas_data, test_size=0.2, random_state=random_state)
        val, test = train_test_split(test, test_size=0.5, random_state=random_state)
        self.train_df = pd.DataFrame(train, columns=self.pandas_data.columns)
        self.val_df = pd.DataFrame(val, columns=self.pandas_data.columns)
        self.test_df = pd.DataFrame(test, columns=self.pandas_data.columns)

        self.label_encoder_user = pp.LabelEncoder()
        self.label_encoder_item = pp.LabelEncoder()
        self.train_df["user_id_idx"] = self.label_encoder_user.fit_transform(self.train_df['user_id'].values)
        self.train_df["item_id_idx"] = self.label_encoder_item.fit_transform(self.train_df['item_id'].values)

        train_user_ids = self.train_df['user_id'].unique()
        train_item_ids = self.train_df['item_id'].unique()

        self.val_df = self.val_df[
            (self.val_df['user_id'].isin(train_user_ids)) & \
            (self.val_df['item_id'].isin(train_item_ids))
            ]

        self.val_df['user_id_idx'] = self.label_encoder_user.transform(self.val_df['user_id'].values)
        self.val_df['item_id_idx'] = self.label_encoder_item.transform(self.val_df['item_id'].values)

        self.test_df = self.test_df[
            (self.test_df['user_id'].isin(train_user_ids)) & \
            (self.test_df['item_id'].isin(train_item_ids))
            ]

        self.test_df['user_id_idx'] = self.label_encoder_user.transform(self.test_df['user_id'].values)
        self.test_df['item_id_idx'] = self.label_encoder_item.transform(self.test_df['item_id'].values)


    def setup(self):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = MovieLensDataset(self.train_df)
        self.val_dataset = MovieLensDataset(self.val_df)
        self.test_dataset = MovieLensDataset(self.test_df)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)



class MovieLensDataModule_CF(MovieLensDataModule):
    def __init__(self, ratings_file, sep='::', batch_size=32, random_state=42):
        super().__init__(ratings_file, sep, batch_size, random_state)

    def setup(self):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = MovieLensDataset(self.train_df)
        self.val_dataset = MovieLensDataset(self.val_df)
        self.test_dataset = MovieLensDataset(self.test_df)

        raw_embeds, embed_layers_config = self.build_embeds(self.train_dataset.num_users, self.train_dataset.num_items)
        self.train_dataset.raw_embeds = raw_embeds
        self.train_dataset.embed_layers_config = embed_layers_config


    def build_embeds(self, num_u, num_v):

        # process users
        embeds_users = np.zeros((num_u, 3))
        cols_users = ['idx', 'sex', 'age_group' ,'occupation', 'zipcode']
        data_users = pd.read_csv('data/ml-1m/users.dat', sep='::', names=cols_users, engine='python')
        
        num_unique_users = []
        for col in ['sex', 'age_group', 'occupation']:
            encoder = pp.LabelEncoder()
            data_users[col] = encoder.fit_transform(data_users[col])
            num_unique_users.append(encoder.classes_.shape[0])

        user_vectors = data_users[['sex', 'age_group', 'occupation']].values
        embeds_users[np.arange(num_u)] = user_vectors[self.label_encoder_user.inverse_transform(np.arange(num_u)) - 1]
        embeds_users = torch.LongTensor(embeds_users)

        # process items
        genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
	              'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        embeds_items = np.zeros((num_v, 1 + len(genres)))

        cols_items = ['idx', 'name', 'genres']
        data_items = pd.read_csv('data/ml-1m/movies.dat', sep='::', names=cols_items, engine='python')
        idx_encoder = pp.LabelEncoder().fit(data_items.idx)

        years_encoder = pp.LabelEncoder()
        years = data_items.name.apply(lambda x: int(x[x.rfind('(')+1:-1]))
        years = years_encoder.fit_transform(years.apply(lambda x: (x//10 * 10) + 5 * (x%10 // 5)))

        genre_encoder = pp.LabelEncoder().fit(genres)

        genres_embed = np.stack(data_items.genres.apply(lambda x: np.where(np.isin(
            np.arange(len(genres)), genre_encoder.transform(x.split('|'))
            ), 1, 0)).values)
        
        item_embeds = np.hstack([years.reshape(-1, 1), genres_embed])
        embeds_items[np.arange(num_v)] = item_embeds[idx_encoder.transform(self.label_encoder_item.inverse_transform(np.arange(num_v)))]
        embeds_items = torch.LongTensor(embeds_items)

        num_unique_items = [years_encoder.classes_.shape[0], len(genres)]

        return {'users': embeds_users, 'items': embeds_items}, {'users': num_unique_users, 'items': num_unique_items}