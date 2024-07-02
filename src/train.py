import click
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

MODELS = {
    "EXSheafGCN": EXSheafGCN,
    "ESheafGCN": ESheafGCN,
    "BimodalSheafGCN": BimodalSheafGCN,
    "BimodalEXSheafGCN": BimodalEXSheafGCN,
    "LightGCN": LightGCN,
    "GAT": GAT
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
@click.option("--dataset_dir", default="../data", type=str)
@click.option("--batch_size", default=1024, type=int)
@click.option("--epochs", default=20, type=int)
def main(model, dataset, latent_dim, dataset_dir, batch_size, epochs):
    model_class = MODELS[model]
    dataset_class, dataset_path = DATASETS[dataset]

    dataset_path = str(pathlib.Path(dataset_dir).joinpath(dataset_path))
    ml_data_module = dataset_class(dataset_path, batchsize=batch_size)
    ml_data_module.setup()
    serialize_dataset(f"DATA_{model}_{dataset}.pickle", ml_data_module)

    train_dataloader = ml_data_module.train_dataloader()
    model = model_class(latent_dim=latent_dim, dataset=ml_data_module.train_dataset)
    trainer = Trainer(max_epochs=epochs, log_every_n_steps=1)
    trainer.fit(model, train_dataloader)
    trainer.save_checkpoint(f"MODEL_{model}_{dataset}.pickle")


if __name__ == "__main__":
    main()

