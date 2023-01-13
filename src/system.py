
import importlib
import pytorch_lightning as pl
import torch.optim as optim


class CustomSystem(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = getattr(
            importlib.import_module(f"models.{config['model']}"),
            f"{config['model_name']}"
        )()
        self.loss = getattr(
            importlib.import_module(f"loss_functions.{config['loss']}"),
            f"{config['loss_name']}"
        )()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.005
        )

    def training_step(self, batch, batch_idx):
        loss = None

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = None

        self.log("val_loss", loss)

    def configure_optimizers(self):
        return self.optimizer
