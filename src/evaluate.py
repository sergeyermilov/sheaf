import torch
import pickle
import pandas as pd

from tqdm import tqdm
from math import log

from src.models.ESheafGCN import ESheafGCN
from src.models.SheafGCN import SheafGCN
from src.models.LightGCN import GCN

pd.set_option('display.max_columns', None)
tqdm.pandas()


def evaluate(user_idx, final_user_Embed, final_item_Embed):
    user_emb = final_user_Embed[user_idx]
    scores = torch.matmul(user_emb, torch.transpose(final_item_Embed, 0, 1))
    return scores.argsort(descending=True).numpy()[:20]


def dcg_at_k(r, k):
    r = r[:k]
    return sum([r[i] / log(i + 2, 2) for i in range(len(r))])

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.0
    return dcg_at_k(r, k) / idcg

# Recommended movies ID

if __name__ == "__main__":
    FILE_NAME = "../data/ml-1m/ratings.dat"
    with open('dataset.pickle', 'rb') as handle:
        ml_data_module = pickle.load(handle)
    ml_1m_train = ml_data_module.train_dataset
    ml_1m_test = ml_data_module.test_dataset
    CHECKPOINT_PATH = "lightning_logs/version_14/checkpoints/epoch=27-step=168.ckpt"
#    CHECKPOINT_PATH = "gcn.ckpt"
    model = GCN.load_from_checkpoint(CHECKPOINT_PATH, dataset=ml_1m_train, latent_dim=64)
    model.eval()
    with torch.no_grad():
        emb0, xmap = model(ml_1m_train.train_edge_index)
        final_user_Embed, final_item_Embed = torch.split(emb0, (ml_1m_train.num_users, ml_1m_train.num_items))

    res = ml_1m_test.interacted_items_by_user_idx.copy(deep=True)
    res["reco"] = res.progress_apply(lambda x: evaluate(x["user_id_idx"], final_user_Embed, final_item_Embed), axis=1)
    res["intersected"] = res.apply(lambda x: list(set(x["reco"]).intersection(x["item_id_idx"])), axis=1)

    res["recall"] = res.apply(lambda x: len(x["intersected"]) / len(x["item_id_idx"]), axis=1)
    res["precision"] = res.apply(lambda x: len(x["intersected"]) / 20, axis=1)

    res[f'ranks_20'] = res.apply(lambda x: [int(movie_id in x["reco"]) for movie_id in x["item_id_idx"]], axis=1)
    k = 20
    res["ndcg_20"] = res.apply(lambda x: ndcg_at_k(x["ranks_20"], k), axis=1)

    print(res["recall"].mean(), res["precision"].mean(), res["ndcg_20"].mean())
    print(res)
    print(res["reco"])
