import torch
from torchmetrics import Metric


class CustomMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("metric", default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, preds, target):
        diff = torch.argmax(preds, dim=1) - torch.argmax(target, dim=1)
        self.metric += diff

    def compute(self):
        return self.metric