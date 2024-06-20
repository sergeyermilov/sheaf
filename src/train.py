import pickle
from pytorch_lightning import Trainer

from src.datasets.movie_lens_1m import MovieLensDataModule
from src.models.ESheafGCN import ESheafGCN, ESheafGCN_content_features
from src.models.LightGCN import GCN

if __name__ == "__main__":

    CONTENT_FEATURE = True

    FILE_NAME = "data/ml-1m/ratings.dat"

    # ml_data_module = MovieLensDataModule(FILE_NAME, batch_size=1024, content_feature=CONTENT_FEATURE)
    # ml_data_module.setup()
    # with open('dataset.pickle', 'wb') as handle:
    #     pickle.dump(ml_data_module, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('dataset.pickle', 'rb') as handle:
        ml_data_module = pickle.load(handle)


    train_dataloader = ml_data_module.train_dataloader()

    if not CONTENT_FEATURE:
        model = ESheafGCN(latent_dim=40, dataset=ml_data_module.train_dataset)
    else:
        model = ESheafGCN_content_features(latent_dim=40, dataset=ml_data_module.train_dataset)
    # model = GCN(latent_dim=64, dataset=ml_data_module.train_dataset)
    trainer = Trainer(max_epochs=5, log_every_n_steps=10)
    trainer.fit(model, train_dataloader)
    trainer.save_checkpoint("gcn.ckpt")

   