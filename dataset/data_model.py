import torch
import random
from torch.utils.data import Dataset, DataLoader


class ContrastiveDataset(Dataset):
    def __init__(self, dataset, model, device):
        self.dataset = dataset
        self.labels_to_indices = self._group_by_label()
        self.model = model
        self.device = device

    def _group_by_label(self):
        labels_to_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in labels_to_indices:
                labels_to_indices[label] = []
            labels_to_indices[label].append(idx)
        return labels_to_indices

    def _find_hard_negative(self, anchor_img, anchor_label):
        different_labels = list(self.labels_to_indices.keys())
        different_labels.remove(anchor_label)
        hardest_negative = None
        min_distance = float("inf")

        for neg_label in different_labels:
            for neg_idx in self.labels_to_indices[neg_label]:
                neg_img, _ = self.dataset[neg_idx]

                with torch.no_grad():
                    anchor_emb = self.model.forward_image(anchor_img.to(self.device).unsqueeze(0))
                    neg_emb = self.model.forward_image(neg_img.to(self.device).unsqueeze(0))
                    distance = torch.dist(anchor_emb, neg_emb).item()

                if distance < min_distance:
                    min_distance = distance
                    hardest_negative = neg_idx

        return hardest_negative

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        get_same_class = random.randint(0, 1)
        if get_same_class:
            idx2 = random.choice(self.labels_to_indices[label1])
        else:
            idx2 = self._find_hard_negative(img1, label1)

        img2, label2 = self.dataset[idx2]
        return img1, img2, torch.tensor([int(label1 != label2)], dtype=torch.float32)

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=True)


class TripletDataset(Dataset):
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
        anchor_img, anchor_label = self.dataset[index]

        positive_idx = index
        while positive_idx == index:
            positive_idx = random.choice(self.labels_to_indices[anchor_label])
        positive_img, _ = self.dataset[positive_idx]

        negative_label = random.choice([lbl for lbl in self.labels_to_indices if lbl != anchor_label])
        negative_idx = random.choice(self.labels_to_indices[negative_label])
        negative_img, _ = self.dataset[negative_idx]

        return anchor_img, positive_img, negative_img

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=True)
