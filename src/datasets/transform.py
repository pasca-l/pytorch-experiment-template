import torch
from torchvision import transforms


class CustomDataTransform():
    def __init__(self) -> None:
        pass

    def __call__(self):
        transform = transforms.Compose([
            transforms.Lambda(
                lambda x: torch.as_tensor(x, dtype=torch.float)
            ),
            transforms.Normalize([0.45], [0.225])
        ])

        return transform
