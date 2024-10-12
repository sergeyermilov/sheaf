import pytorch_lightning as pl


class TopKPopularity(pl.LightningModule):
    def __init__(self,
                 dataset):
        super(TopKPopularity, self).__init__()
        self.dataset = dataset
        self.top = self.dataset.pandas_data.groupby('item_id_idx').size().sort_values(ascending=False).index.values

    def forward(self):
        return self.top

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        pass

    def evaluate(self, interacted_user_id_idx, k):
        res = []
        interacted_set = set(interacted_user_id_idx)
        for item in self.top:
            if item not in interacted_set:
                res.append(item)
            if len(res) == k:
                return res
        return []