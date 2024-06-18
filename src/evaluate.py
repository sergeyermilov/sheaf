import numpy as np
import torch
import pickle
import pandas as pd

from tqdm import tqdm
from math import log

from src.models.ESheafGCN import ESheafGCN
from src.models.GAT import GAT

pd.set_option('display.max_columns', None)
tqdm.pandas()


def evaluate(user_idx, final_user_Embed, final_item_Embed, interacted, k):
    user_emb = final_user_Embed[user_idx]
    scores = torch.matmul(user_emb, torch.transpose(final_item_Embed, 0, 1))
    scores[interacted] = 0.0
    return scores.argsort(descending=True).numpy()[:k]


def dcg_at_k(score, k=None):
    if k is not None:
        score = score[:k]

    discounts = np.log2(np.arange(2, len(score) + 2))
    dcg = np.sum(score / discounts)
    return dcg


def ndcg_at_k(score, k=None):
    actual_dcg = dcg_at_k(score, k)
    sorted_score = np.sort(score)[::-1]
    best_dcg = dcg_at_k(sorted_score, k)
    ndcg = actual_dcg / best_dcg
    return ndcg

def get_metrics(_df, k):
    _df[f"reco_{k}"] = _df.progress_apply(lambda x: evaluate(x["user_id_idx"], final_user_Embed, final_item_Embed, x["interacted_id_idx"], k), axis=1)
    _df[f"intersected_{k}"] = _df.apply(lambda x: list(set(x[f"reco_{k}"]).intersection(x["item_id_idx"])), axis=1)

    _df[f"recall_{k}"] = _df.apply(lambda x: len(x[f"intersected_{k}"]) / len(x["item_id_idx"]), axis=1)
    _df[f"precision_{k}"] = _df.apply(lambda x: len(x[f"intersected_{k}"]) / k, axis=1)

    _df[f'ranks_{k}'] = _df.apply(lambda x: [int(movie_id in x[f"reco_{k}"]) for movie_id in x["item_id_idx"]], axis=1)
    _df[f"ndcg_{k}"] = _df.apply(lambda x: ndcg_at_k(x[f'ranks_{k}'], k), axis=1)
    return _df


# Recommended movies ID

if __name__ == "__main__":
    with open('dataset.pickle', 'rb') as handle:
        ml_data_module = pickle.load(handle)
    ml_1m_train = ml_data_module.train_dataset
    ml_1m_test = ml_data_module.test_dataset
    CHECKPOINT_PATH = "gcn.ckpt"
    model = ESheafGCN.load_from_checkpoint(CHECKPOINT_PATH, dataset=ml_1m_train, latent_dim=40)
    model.eval()
    with torch.no_grad():
        emb0, xmap, y, out, _, _, _ = model(ml_1m_train.adjacency_matrix)
        final_user_Embed, final_item_Embed = torch.split(out, (ml_1m_train.num_users, ml_1m_train.num_items))

    res = ml_1m_test.interacted_items_by_user_idx.copy(deep=True)
    interactions = ml_1m_train.interacted_items_by_user_idx.copy(deep=True).rename(columns={"item_id_idx": "interacted_id_idx"})
    res = res.merge(interactions, on=["user_id_idx"])

    res = get_metrics(res, 20)
    res = get_metrics(res, 10)

    print(res["recall_20"].mean(), res["precision_20"].mean(), res["ndcg_20"].fillna(0.0).mean())
    print(res["recall_10"].mean(), res["precision_10"].mean(), res["ndcg_10"].fillna(0.0).mean())