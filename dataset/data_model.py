import torch
from torch.utils.data import Dataset, DataLoader
import random


class ContrastiveDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]

        get_same_class = random.randint(0, 1)
        while True:
            new_idx = random.randint(0, len(self.dataset) - 1)
            new_imag, new_label = self.dataset[new_idx]
            if get_same_class and new_label == label:
                break
            elif get_same_class == 0 and new_label != label:
                break
        return img, new_imag, torch.tensor([int(label != new_label)], dtype=torch.float32)

    def get_dataloader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
