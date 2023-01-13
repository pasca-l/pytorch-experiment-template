import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrained = False
        self.model = nn.Sequential(
            nn.Linear(1000, 100)
        )

    def forward(self, x):
        return self.model(x)
