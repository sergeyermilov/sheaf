import os
import click
import torch
import pickle
import pathlib

from pytorch_lightning import Trainer

from src.datasets.facebook import FacebookDataModule, FACEBOOK_DATASET_RELATIVE_PATH
from src.datasets.movie_lens_1m import MovieLensDataModule, MOVIE_LENS_DATASET_RELATIVE_PATH
from src.datasets.yahoo_movies import YahooMoviesDataModule, YAHOO_DATASET_RELATIVE_PATH

from src.models.EXSheafGCN import EXSheafGCN
from src.models.ESheafGCN import ESheafGCN
from src.models.BimodalSheafGCN import BimodalSheafGCN
from src.models.BimodalEXSheafGCN import BimodalEXSheafGCN
from src.models.LightGCN import LightGCN
from src.models.GAT import GAT
from src.models.SheafGCN import SheafGCN

MODELS = {
    "EXSheafGCN": EXSheafGCN,
    "ESheafGCN": ESheafGCN,
    "BimodalSheafGCN": BimodalSheafGCN,
    "BimodalEXSheafGCN": BimodalEXSheafGCN,
    "LightGCN": LightGCN,
    "GAT": GAT,
    "SheafGCN": SheafGCN,
}

DATASETS = {
    "FACEBOOK": (FacebookDataModule, FACEBOOK_DATASET_RELATIVE_PATH),
    "MOVIELENS": (MovieLensDataModule, MOVIE_LENS_DATASET_RELATIVE_PATH),
    "YAHOO": (YahooMoviesDataModule, YAHOO_DATASET_RELATIVE_PATH)
}

def serialize_dataset(filename, datamodule):
    with open(filename, 'wb') as handle:
        pickle.dump(datamodule, handle, protocol=pickle.HIGHEST_PROTOCOL)

@click.command()
@click.option("--model", default="LightGCN", type=str)
@click.option("--dataset", default="FACEBOOK", type=str)
@click.option("--latent_dim", default=40, type=int)
@click.option("--dataset_dir", default="data/", type=str)
@click.option("--batch_size", default=1024, type=int)
@click.option("--epochs", default=20, type=int)
@click.option("--device", default="cuda", type=str)
@click.option("--artifact_dir", default="artifact/", type=str)
def main(model, dataset, latent_dim, dataset_dir, batch_size, epochs, device, artifact_dir):
    print("-----------------------------------------------")
    print("Running model with the following configuration:")
    print(f"model = {model}")
    print(f"dataset = {dataset}")
    print(f"latent_dim = {latent_dim}")
    print(f"dataset_dir = {dataset_dir}")
    print(f"batch_size = {batch_size}")
    print(f"epochs = {epochs}")
    print(f"device = {device}")
    print("-----------------------------------------------")
    artifact_dir = pathlib.Path(artifact_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    if os.getenv("CUDA_VISIBLE_DEVICE"):
        raise Exception("You need to fix CUDA_VISIBLE_DEVICE to desired device. Distributed training is not yet supported.")

    torch.set_default_device(device)

    model_class = MODELS[model]
    dataset_class, dataset_path = DATASETS[dataset]

    dataset_path = str(pathlib.Path(dataset_dir).joinpath(dataset_path))
    ml_data_module = dataset_class(dataset_path, batch_size=batch_size)
    ml_data_module.setup()

    serialize_dataset(str(artifact_dir.joinpath(f"DATA_{model}_{dataset}_{epochs}.pickle")), ml_data_module)

    train_dataloader = ml_data_module.train_dataloader()
    model_instance = model_class(latent_dim=latent_dim, dataset=ml_data_module.train_dataset)
    trainer = Trainer(max_epochs=epochs, log_every_n_steps=1)
    trainer.fit(model_instance, train_dataloader)
    trainer.save_checkpoint(str(artifact_dir.joinpath(f"MODEL_{model}_{dataset}_{epochs}.pickle")))


if __name__ == "__main__":
    main()

   