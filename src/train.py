import pickle
from pytorch_lightning import Trainer

from src.datasets.movie_lens_1m import MovieLensDataModule
from src.models.ESheafGCN import ESheafGCN, ESheafGCN_wo_embed
from src.models.LightGCN import GCN

if __name__ == "__main__":

    TRAIN_EMBEDS = False

    FILE_NAME = "data/ml-1m/ratings.dat"

    ml_data_module = MovieLensDataModule(FILE_NAME, batch_size=1024, learn_embeds=TRAIN_EMBEDS, embed_size=40)
    ml_data_module.setup()
    with open('dataset.pickle', 'wb') as handle:
        pickle.dump(ml_data_module, handle, protocol=pickle.HIGHEST_PROTOCOL)


    train_dataloader = ml_data_module.train_dataloader()
    model = ESheafGCN_wo_embed(latent_dim=40, dataset=ml_data_module.train_dataset, learn_embeds=TRAIN_EMBEDS)
    # model = GCN(latent_dim=64, dataset=ml_data_module.train_dataset)
    trainer = Trainer(max_epochs=5, log_every_n_steps=10)
    trainer.fit(model, train_dataloader)
    trainer.save_checkpoint("gcn.ckpt")

   