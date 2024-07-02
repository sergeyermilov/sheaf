import os
import torch
import click
import json
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.EXSheafGCN import EXSheafGCN
from src.models.ESheafGCN import ESheafGCN
from src.models.BimodalSheafGCN import BimodalSheafGCN
from src.models.BimodalEXSheafGCN import BimodalEXSheafGCN
from src.models.LightGCN import LightGCN
from src.models.GAT import GAT

MODELS = {
    "EXSheafGCN": EXSheafGCN,
    "ESheafGCN": ESheafGCN,
    "BimodalSheafGCN": BimodalSheafGCN,
    "BimodalEXSheafGCN": BimodalEXSheafGCN,
    "LightGCN": LightGCN,
    "GAT": GAT
}

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

def get_metrics(_df, k, user_embeddings, item_embeddings):
    _df[f"reco_{k}"] = _df.progress_apply(lambda x: evaluate(x["user_id_idx"], user_embeddings, item_embeddings, x["interacted_id_idx"], k), axis=1)
    _df[f"intersected_{k}"] = _df.apply(lambda x: list(set(x[f"reco_{k}"]).intersection(x["item_id_idx"])), axis=1)

    _df[f"recall_{k}"] = _df.apply(lambda x: len(x[f"intersected_{k}"]) / len(x["item_id_idx"]), axis=1)
    _df[f"precision_{k}"] = _df.apply(lambda x: len(x[f"intersected_{k}"]) / k, axis=1)

    _df[f'ranks_{k}'] = _df.apply(lambda x: [int(movie_id in x[f"reco_{k}"]) for movie_id in x["item_id_idx"]], axis=1)
    _df[f"ndcg_{k}"] = _df.apply(lambda x: ndcg_at_k(x[f'ranks_{k}'], k), axis=1)
    return _df


@click.command()
@click.option("--model", default="LightGCN", type=str)
@click.option("--dataset", default="LightGCN", type=str)
@click.option("--epochs", default=5, type=int)
@click.option("--artifact_dir", default="artifact/", type=str)
def main(model, dataset, epochs, artifact_dir):
    print("-----------------------------------------------")
    print("Running model with the following configuration:")
    print(f"model = {model}")
    print(f"dataset = {dataset}")
    print(f"epochs = {epochs}")
    print(f"artifact_dir = {artifact_dir}")
    print("-----------------------------------------------")

    if os.getenv("CUDA_VISIBLE_DEVICE"):
        raise Exception("You need to fix CUDA_VISIBLE_DEVICE to desired device. Distributed training is not yet supported.")

    with open(f"DATA_{model}_{dataset}_{epochs}.pickle", 'rb') as handle:
        ml_data_module = pickle.load(handle)

    ml_1m_train = ml_data_module.train_dataset
    ml_1m_test = ml_data_module.test_dataset

    model = MODELS[model].load_from_checkpoint(f"MODEL_{model}_{dataset}_{epochs}.pickle", dataset=ml_1m_train, latent_dim=40)
    model.eval()

    with torch.no_grad():
        _, embeddings = model(ml_1m_train.adjacency_matrix)
        user_embeddings, item_embeddings = torch.split(embeddings, (ml_1m_train.num_users, ml_1m_train.num_items))

    res = ml_1m_test.interacted_items_by_user_idx.copy(deep=True)
    interactions = ml_1m_train.interacted_items_by_user_idx.copy(deep=True).rename(
        columns={"item_id_idx": "interacted_id_idx"})
    res = res.merge(interactions, on=["user_id_idx"])

    res = get_metrics(res, 5, user_embeddings, item_embeddings)
    res = get_metrics(res, 10, user_embeddings, item_embeddings)
    res = get_metrics(res, 20, user_embeddings, item_embeddings)
    res = get_metrics(res, 50, user_embeddings, item_embeddings)

    res.to_csv(f"DETAILED_{model}_{dataset}_{epochs}.csv")

    brief = dict()
    for c in res.columns:
        brief[c] = res[c].mean()

    with open(f"BRIEF_{model}_{dataset}_{epochs}.json", "w") as brief_file:
        json.dump(brief, brief_file)

    print(f"Evaluation results for model {model} over dataset {dataset} that was trained on {epochs} epochs:")
    for k, v in brief.items():
        print(f"{k}: {v}")
