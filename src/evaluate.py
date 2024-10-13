import os
import typing
from functools import partial

import torch
import click
import json
import pickle
import pathlib
import datetime
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.best.ease import EASE
from src.models.best.top import TopKPopularity
from src.models.sheaf.ExtendableSheafGCN import ExtendableSheafGCN
from src.models.sheaf.FastESheafGCN import FastESheafGCN
from src.models.sheaf.SheafGCN import SheafGCN
from src.models.sheaf.ESheafGCN import ESheafGCN

from src.models.graph.LightGCN import LightGCN
from src.models.graph.GAT import GAT

MODELS = {
    # sheaf models
    "ExtendableSheafGCN": ExtendableSheafGCN,
    "ESheafGCN": ESheafGCN,
    "SheafGCN": SheafGCN,
    # other models
    "LightGCN": LightGCN,
    "FastESheafGCN": FastESheafGCN,
    "GAT": GAT,
    "TopKPopularity": TopKPopularity,
    "EASE": EASE
}


def as_numpy(torch_tensor):
    if torch_tensor.device == "cpu":
        return torch_tensor.numpy()

    return torch_tensor.cpu().numpy()


def infer_dotprod(user_idx, users_embed, items_embed, interacted, k):
    scores = torch.matmul(users_embed[user_idx], torch.transpose(items_embed, 0, 1))
    scores[interacted] = -torch.inf
    return as_numpy(scores.argsort(descending=True))[:k].tolist()


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


def get_metrics(_df: pd.DataFrame, k: int, compute_recs_fn: typing.Callable):
    reco_k = f"reco_{k}"
    intersected_k = f"intersected_{k}"
    recall_k = f"recall_{k}"
    precision_k = f"precision_{k}"
    ranks_k = f"ranks_{k}"
    ndcg_k = f"ndcg_{k}"

    _df[reco_k] = _df.progress_apply(partial(compute_recs_fn, k=k), axis=1)

    _df[intersected_k] = _df.apply(lambda x: list(set(x[f"reco_{k}"]).intersection(x["item_id_idx"])), axis=1)

    _df[recall_k] = _df.apply(lambda x: len(x[f"intersected_{k}"]) / len(x["item_id_idx"]), axis=1)
    _df[precision_k] = _df.apply(lambda x: len(x[f"intersected_{k}"]) / k, axis=1)

    _df[ranks_k] = _df.apply(lambda x: [int(reco_idx in x["item_id_idx"]) for reco_idx in x[f"reco_{k}"]], axis=1)
    _df[ndcg_k] = _df.apply(lambda x: ndcg_at_k(x[f'ranks_{k}'], k), axis=1).fillna(0.0)

    return _df, [recall_k, precision_k, ndcg_k]


@click.command()
@click.option("--device", default="cuda", type=str)
@click.option("--artifact-id", type=str, required=True)
@click.option("--artifact-dir", default="artifact/", type=pathlib.Path)
@click.option("--model-name", default="model.pickle", type=str)
def main(device, artifact_id, artifact_dir, model_name):
    artifact_dir = artifact_dir.joinpath(artifact_id)

    with open(artifact_dir.joinpath("config.json"), "r") as fhandle:
        configs = json.load(fhandle)

    model = configs['model']
    dataset = configs['dataset']
    model_params = json.loads(configs['model_params'].replace("'", "\""))
    dataset_params = json.loads(configs['dataset_params'].replace("'", "\""))
    epochs = configs['epochs']
    denoise = configs['denoise']

    print("------------------------------------------------")
    print("Evaluate model with the following configuration:")
    print("------------------------------------------------")
    print(f"date= {datetime.datetime.now()}")
    print(f"model = {model}")
    print(f"dataset = {dataset}")
    print(f"model_name = {model_name}")
    print(f"model-params = {model_params}")
    print(f"dataset-params = {dataset_params}")
    print(f"epochs = {epochs}")
    print(f"device = {device}")
    print(f"artifact-dir = {artifact_dir}")
    print(f"denoise = {denoise}")
    print("------------------------------------------------")

    if os.getenv("CUDA_VISIBLE_DEVICE"):
        raise Exception(
            "You need to fix CUDA_VISIBLE_DEVICE to desired device. Distributed training is not yet supported.")

    pd.set_option('display.max_columns', None)
    tqdm.pandas()

    torch.set_default_device(device)

    os.makedirs(artifact_dir, exist_ok=True)

    with open(artifact_dir.joinpath(f"data.pickle"), 'rb') as handle:
        ml_data_module = pickle.load(handle)

    train_dataset = ml_data_module.train_dataset
    test_dataset = ml_data_module.test_dataset

    model_instance = MODELS[model].load_from_checkpoint(
        artifact_dir.joinpath(f"{model_name}"), dataset=train_dataset, **model_params
    )
    model_instance.eval()
    model_instance = model_instance.to(device)

    res = test_dataset.interacted_items_by_user_idx.copy(deep=True).reset_index()
    interactions = train_dataset.interacted_items_by_user_idx.copy(deep=True).reset_index().rename(
        columns={"item_id_idx": "interacted_id_idx"})
    res = res.merge(interactions, on=["user_id_idx"])

    if not hasattr(model, "evaluate"):
        with torch.no_grad():
            if denoise:
                embeddings = model_instance.get_denoised_embeddings()
            else:
                _, embeddings = model_instance(train_dataset.train_edge_index.to(device))

            user_embeddings, item_embeddings = torch.split(
                embeddings, [test_dataset.num_users, test_dataset.num_items]
            )

            compute_recs_fn = lambda x, k: infer_dotprod(x["user_id_idx"], user_embeddings, item_embeddings, x["interacted_id_idx"], k)
    else:
        compute_recs_fn = lambda x, k: model.evaluate(x["interacted_id_idx"], k)

    res, metrics_5 = get_metrics(res, 5, compute_recs_fn)
    res, metrics_10 = get_metrics(res, 10, compute_recs_fn)
    res, metrics_20 = get_metrics(res, 20, compute_recs_fn)
    res, metrics_50 = get_metrics(res, 50, compute_recs_fn)

    os.makedirs(artifact_dir.joinpath(f"reports"), exist_ok=True)

    brief = dict()
    for c in itertools.chain(metrics_5, metrics_10, metrics_20, metrics_50):
        brief[c] = res[c].mean()

    with open(artifact_dir.joinpath(f"reports").joinpath(f"{model_name}_report.json"), "w") as brief_file:
        json.dump(brief, brief_file)

    print(f"Evaluation results for model {model} over dataset {dataset} that was trained on {epochs} epochs:")
    for k, v in brief.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
