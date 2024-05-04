import pickle
from pytorch_lightning import Trainer

from src.datasets.movie_lens_1m import MovieLensDataModule
from src.models.ESheafGCN import ESheafGCN
from src.models.LightGCN import GCN

if __name__ == "__main__":
    FILE_NAME = "../data/ml-1m/ratings.dat"
    ml_data_module = MovieLensDataModule(FILE_NAME, batch_size=1024)
    ml_data_module.setup()
    with open('dataset.pickle', 'wb') as handle:
        pickle.dump(ml_data_module, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_dataloader = ml_data_module.train_dataloader()
 #   model = ESheafGCN(latent_dim=40, dataset=ml_data_module.train_dataset)
    model = GCN(latent_dim=64, dataset=ml_data_module.train_dataset)
    trainer = Trainer(max_epochs=100, log_every_n_steps=1)
    trainer.fit(model, train_dataloader)
    trainer.save_checkpoint("gcn.ckpt")

