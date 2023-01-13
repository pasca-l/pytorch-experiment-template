import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import CustomDataset
from transform import CustomDataTransform


class CustomDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = CustomDataset(
                data_dir="/path/to/data",
                ann="/path/to/annotation",
                transform=CustomDataTransform,
            )
            self.val_data = None

        if stage == "predict":
            self.predict_data = None

    def train_dataloader(self):
        return DataLoader(
            self.train_data
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data
        )
