import os
import torch
import click
import json
import pickle
import pathlib
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.sheaf.GEXSheafGCN import GEXSheafGCN
from src.models.sheaf.XSheafGCN import XSheafGCN
from src.models.sheaf.GXSheafGCN import GXSheafGCN
from src.models.sheaf.GSheafGCN import GSheafGCN
from src.models.sheaf.ESheafGCN import ESheafGCN
from src.models.sheaf.SheafGCN import SheafGCN

from src.models.LightGCN import LightGCN
from src.models.GAT import GAT

MODELS = {
    "GEXSheafGCN": GEXSheafGCN,
    "ESheafGCN": ESheafGCN,
    "XSheafGCN": XSheafGCN,
    "GXSheafGCN": GXSheafGCN,
    "GSheafGCN": GSheafGCN,
    "LightGCN": LightGCN,
    "GAT": GAT,
    "SheafGCN": SheafGCN,
}


def as_numpy(torch_tensor):
    if torch_tensor.device == "cpu":
        return torch_tensor.numpy()

    return torch_tensor.cpu().numpy()


def evaluate(user_idx, final_user_Embed, final_item_Embed, interacted, k):
    user_emb = final_user_Embed[user_idx]
    scores = torch.matmul(user_emb, torch.transpose(final_item_Embed, 0, 1))
    scores[interacted] = 0.0
    return as_numpy(scores.argsort(descending=True))[:k]


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
    reco_k = f"reco_{k}"
    intersected_k = f"intersected_{k}"
    recall_k = f"recall_{k}"
    precision_k = f"precision_{k}"
    ranks_k = f'ranks_{k}'
    ndcg_k = f"ndcg_{k}"

    _df[reco_k] = _df.progress_apply(
        lambda x: evaluate(x["user_id_idx"], user_embeddings, item_embeddings, x["interacted_id_idx"], k), axis=1)
    _df[intersected_k] = _df.apply(lambda x: list(set(x[f"reco_{k}"]).intersection(x["item_id_idx"])), axis=1)

    _df[recall_k] = _df.apply(lambda x: len(x[f"intersected_{k}"]) / len(x["item_id_idx"]), axis=1)
    _df[precision_k] = _df.apply(lambda x: len(x[f"intersected_{k}"]) / k, axis=1)

    _df[ranks_k] = _df.apply(lambda x: [int(movie_id in x[f"reco_{k}"]) for movie_id in x["item_id_idx"]], axis=1)
    _df[ndcg_k] = _df.apply(lambda x: ndcg_at_k(x[f'ranks_{k}'], k), axis=1).fillna(0.0)

    return _df, [recall_k, precision_k, ndcg_k]


@click.command()
@click.option("--model", default="LightGCN", type=str)
@click.option("--dataset", default="LightGCN", type=str)
@click.option("--epochs", default=5, type=int)
@click.option("--artifact_dir", default="artifact/", type=str)
@click.option("--report_dir", default="report/", type=str)
@click.option("--device", default="cuda", type=str)
def main(model, dataset, epochs, artifact_dir, report_dir, device):
    print("-----------------------------------------------")
    print("Running model with the following configuration:")
    print(f"model = {model}")
    print(f"dataset = {dataset}")
    print(f"epochs = {epochs}")
    print(f"artifact_dir = {artifact_dir}")
    print(f"device = {device}")
    print("-----------------------------------------------")

    if os.getenv("CUDA_VISIBLE_DEVICE"):
        raise Exception(
            "You need to fix CUDA_VISIBLE_DEVICE to desired device. Distributed training is not yet supported.")

    pd.set_option('display.max_columns', None)
    tqdm.pandas()

    torch.set_default_device(device)

    artifact_dir = pathlib.Path(artifact_dir)
    report_dir = pathlib.Path(report_dir)

    os.makedirs(report_dir, exist_ok=True)

    with open(str(artifact_dir.joinpath(f"DATA_{model}_{dataset}_{epochs}.pickle")), 'rb') as handle:
        ml_data_module = pickle.load(handle)

    train_dataset = ml_data_module.train_dataset
    test_dataset = ml_data_module.test_dataset

    model_instance = MODELS[model].load_from_checkpoint(
        str(artifact_dir.joinpath(f"MODEL_{model}_{dataset}_{epochs}.pickle")), dataset=train_dataset, latent_dim=40)
    model_instance.eval()
    model_instance = model_instance.to(device)

    with torch.no_grad():
        _, embeddings = model_instance(train_dataset.adjacency_matrix.to(device))
        user_embeddings, item_embeddings = torch.split(embeddings, (train_dataset.num_users, train_dataset.num_items))

    res = test_dataset.interacted_items_by_user_idx.copy(deep=True)
    interactions = train_dataset.interacted_items_by_user_idx.copy(deep=True).rename(
        columns={"item_id_idx": "interacted_id_idx"})
    res = res.merge(interactions, on=["user_id_idx"])

    res, metrics_5 = get_metrics(res, 5, user_embeddings, item_embeddings)
    res, metrics_10 = get_metrics(res, 10, user_embeddings, item_embeddings)
    res, metrics_20 = get_metrics(res, 20, user_embeddings, item_embeddings)
    res, metrics_50 = get_metrics(res, 50, user_embeddings, item_embeddings)

    res.to_csv(str(report_dir.joinpath(f"DETAILED_{model}_{dataset}_{epochs}.csv")))

    brief = dict()
    for c in itertools.chain(metrics_5, metrics_10, metrics_20, metrics_50):
        brief[c] = res[c].mean()

    with open(str(report_dir.joinpath(f"BRIEF_{model}_{dataset}_{epochs}.json")), "w") as brief_file:
        json.dump(brief, brief_file)

    print(f"Evaluation results for model {model} over dataset {dataset} that was trained on {epochs} epochs:")
    for k, v in brief.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
