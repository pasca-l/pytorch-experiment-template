import torch
import torch.nn as nn


class MeanAbsoluteError(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, outputs, targets):
        loss = torch.mean(torch.abs(outputs - targets))

        return loss
