from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, ann=None, transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.ann = ann
        self.transform = transform

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        return index
