import os
import json
import click
import torch
import pathlib
import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.data.datasets.actor_dataset import ActorDataModule, ACTOR_DATASET_RELATIVE_PATH
from src.data.datasets.chameleon import ChameleonDataModule, CHAMELEON_DATASET_RELATIVE_PATH
from src.data.datasets.cora_full import CoraFullDataModule, CORA_FULL_DATASET_RELATIVE_PATH
from src.data.datasets.texas import TexasDataModule, TEXAS_DATASET_RELATIVE_PATH
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
from src.models.sheaf.FastESheafGCNNodeClassification import FastESheafGCNNodeClassification

from src.models.graph.LightGCN import LightGCN
from src.models.graph.GAT import GAT

MODELS = {
    # sheaf models
    "ExtendableSheafGCN": ExtendableSheafGCN,
    "ESheafGCN": ESheafGCN,
    "FastESheafGCNNodeClassification": FastESheafGCNNodeClassification,
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
    "YAHOO": (YahooMoviesDataModule, YAHOO_DATASET_RELATIVE_PATH),
    "TEXAS": (TexasDataModule, TEXAS_DATASET_RELATIVE_PATH),
    "CHAMELEON":  (ChameleonDataModule, CHAMELEON_DATASET_RELATIVE_PATH),
    "CORA_FULL": (CoraFullDataModule, CORA_FULL_DATASET_RELATIVE_PATH),
    "ACTOR": (ActorDataModule, ACTOR_DATASET_RELATIVE_PATH)
}


@click.command()
@click.option("--model", default="FastESheafGCNNodeClassification", type=str)
@click.option("--dataset", default="TEXAS", type=str)
@click.option("--dataset-params", default="{}", type=str)
@click.option("--model-params", default="""{"latent_dim": 40}""", type=str)
@click.option("--dataset-dir", default="data/", type=pathlib.Path)
@click.option("--epochs", default=30, type=int)
@click.option("--device", default="cpu", type=str)
@click.option("--artifact-dir", default="artifact/", type=pathlib.Path)
@click.option("--denoise", is_flag=True)
def main(model, dataset, dataset_params, model_params, dataset_dir, epochs, device, artifact_dir, denoise):
    artifact_params = dict(
        model=model,
        dataset=dataset,
        epochs=epochs,
        model_params=model_params,
        dataset_params=dataset_params,
        denoise=denoise,
    )

    artifact_id = compute_artifact_id(length=12, **artifact_params)

    print("-----------------------------------------------")
    print("Running model with the following configuration:")
    print("-----------------------------------------------")
    print(f"date= {datetime.datetime.now()}")
    print(f"model = {model}")
    print(f"dataset = {dataset}")
    print(f"dataset-params = {dataset_params}")
    print(f"model-params = {model_params}")
    print(f"dataset-dir = {dataset_dir}")
    print(f"epochs = {epochs}")
    print(f"device = {device}")
    print(f"artifact-dir = {artifact_dir}")
    print(f"artifact-id = {artifact_id}")
    print(f"denoise = {denoise}")
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
    dataset_class_partial = create_from_json_string(dataset_class, dataset_params)

    ml_data_module = dataset_class_partial(str(dataset_dir.joinpath(dataset_path)), device=device)
    ml_data_module.setup()

    serialize_dataset(artifact_dir.joinpath(f"data.pickle"), ml_data_module)

    train_dataloader = ml_data_module.train_dataloader()

    model_class_partial = create_from_json_string(model_class, model_params)
    model_instance = model_class_partial(dataset=ml_data_module.train_dataset)

    if denoise:
        if not hasattr(model_instance, "is_denoisable"):
            raise Exception("Model is not denoisable")

        if not model_instance.is_denoisable():
            raise Exception("Current configuration is not denoisable")

    trainer = Trainer(max_epochs=epochs, log_every_n_steps=1, logger=[
        CSVLogger(artifact_dir, name="train_logs"),
        TensorBoardLogger(artifact_dir, name="train_tb")
    ])
    trainer.fit(model_instance, train_dataloader)
    trainer.save_checkpoint(str(artifact_dir.joinpath(f"model.pickle")))


if __name__ == "__main__":
    main()
