import torch
import random
from torch.utils.data import Dataset, DataLoader


class ContrastiveDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels_to_indices = self._group_by_label()

    def _group_by_label(self):
        labels_to_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in labels_to_indices:
                labels_to_indices[label] = []
            labels_to_indices[label].append(idx)
        return labels_to_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]

        get_same_class = random.randint(0, 1)

        if get_same_class:
            idx2 = random.choice(self.labels_to_indices[label1])
        else:
            different_labels = list(self.labels_to_indices.keys())
            different_labels.remove(label1)
            label2 = random.choice(different_labels)
            idx2 = random.choice(self.labels_to_indices[label2])
        img2, label2 = self.dataset[idx2]
        return img1, img2, torch.tensor([int(label1 != label2)], dtype=torch.float32)

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=True)
