import os
import json
import click
import torch
import pathlib
import datetime

from pytorch_lightning import Trainer

from src.models.best.ease import EASE
from src.models.best.top import TopKPopularity
from src.models.sheaf.FastESheafGCN import FastESheafGCN
from src.utils import create_from_json_string
from src.utils import serialize_dataset
from src.utils import compute_artifact_id

from src.data.datasets.facebook import FacebookDataModule, FACEBOOK_DATASET_RELATIVE_PATH
from src.data.datasets.movie_lens import MovieLensDataModule, MOVIE_LENS_1M_DATASET_RELATIVE_PATH, MOVIE_LENS_10M_DATASET_RELATIVE_PATH
from src.data.datasets.yahoo_movies import YahooMoviesDataModule, YAHOO_DATASET_RELATIVE_PATH

from src.models.sheaf.ExtendableSheafGCN import ExtendableSheafGCN
from src.models.sheaf.SheafGCN import SheafGCN
from src.models.sheaf.ESheafGCN import ESheafGCN

from src.models.graph.LightGCN import LightGCN
from src.models.graph.GAT import GAT

MODELS = {
    # sheaf models
    "ExtendableSheafGCN": ExtendableSheafGCN,
    "ESheafGCN": ESheafGCN,
    "FastESheafGCN": FastESheafGCN,
    "SheafGCN": SheafGCN,
    # graph models
    "LightGCN": LightGCN,
    "GAT": GAT,
    "TopKPopularity": TopKPopularity,
    "EASE": EASE
}

DATASETS = {
    "FACEBOOK": (FacebookDataModule, FACEBOOK_DATASET_RELATIVE_PATH),
    "MOVIELENS1M": (MovieLensDataModule, MOVIE_LENS_1M_DATASET_RELATIVE_PATH),
    "MOVIELENS10M": (MovieLensDataModule, MOVIE_LENS_10M_DATASET_RELATIVE_PATH),
    "YAHOO": (YahooMoviesDataModule, YAHOO_DATASET_RELATIVE_PATH)
}


@click.command()
@click.option("--model", default="LightGCN", type=str)
@click.option("--dataset", default="FACEBOOK", type=str)
@click.option("--split", default="simple", type=click.Choice(['time', 'simple']))
@click.option("--params", default="{}", type=str)
@click.option("--dataset-dir", default="data/", type=pathlib.Path)
@click.option("--batch-size", default=1024, type=int)
@click.option("--epochs", default=20, type=int)
@click.option("--device", default="cuda", type=str)
@click.option("--artifact-dir", default="artifact/", type=pathlib.Path)
def main(model, dataset, split, params, dataset_dir, batch_size, epochs, device, artifact_dir):
    artifact_params = dict(
        model=model,
        dataset=dataset,
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        params=params
    )

    artifact_id = compute_artifact_id(length=12, **artifact_params)

    print("-----------------------------------------------")
    print("Running model with the following configuration:")
    print("-----------------------------------------------")
    print(f"date= {datetime.datetime.now()}")
    print(f"model = {model}")
    print(f"dataset = {dataset}")
    print(f"params = {params}")
    print(f"split = {split}")
    print(f"dataset_dir = {dataset_dir}")
    print(f"batch_size = {batch_size}")
    print(f"epochs = {epochs}")
    print(f"device = {device}")
    print(f"artifact_dir = {artifact_dir}")
    print(f"artifact_id = {artifact_id}")
    print("-----------------------------------------------")

    artifact_dir = artifact_dir.joinpath(artifact_id)
    os.makedirs(artifact_dir, exist_ok=True)

    with open(artifact_dir.joinpath("config.json"), "w") as fhandle:
        json.dump(artifact_params, fhandle)

    if os.getenv("CUDA_VISIBLE_DEVICE"):
        raise Exception(
            "You need to fix CUDA_VISIBLE_DEVICE to desired device. Distributed training is not yet supported.")

    torch.set_default_device(device)

    model_class = MODELS[model]
    dataset_class, dataset_path = DATASETS[dataset]

    dataset_path = str(dataset_dir.joinpath(dataset_path))
    ml_data_module = dataset_class(dataset_path, batch_size=batch_size)
    ml_data_module.setup()

    serialize_dataset(artifact_dir.joinpath(f"data.pickle"), ml_data_module)

    train_dataloader = ml_data_module.train_dataloader()

    model_class_partial = create_from_json_string(model_class, params)
    model_instance = model_class_partial(dataset=ml_data_module.train_dataset)
    trainer = Trainer(max_epochs=epochs, log_every_n_steps=1)
    trainer.fit(model_instance, train_dataloader)
    trainer.save_checkpoint(str(artifact_dir.joinpath(f"model.pickle")))


if __name__ == "__main__":
    main()
