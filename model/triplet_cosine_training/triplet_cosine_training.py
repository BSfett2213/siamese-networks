import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
from dataset.data_model import TripletCosineDataset
from model.loss_types import TripletCosineLoss


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)
        self.normalize = nn.functional.normalize

    def forward(self, x):
        x = self.backbone(x)
        return self.normalize(x, p=2, dim=1)


transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

dataset = TripletCosineDataset("../../dataset/extracted_faces", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
model = SiameseNetwork()
loss_fn = TripletCosineLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for anchor, positive, negative, _, _ in dataloader:
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/10], Loss: {total_loss / len(dataloader):.4f}")
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for anchor, positive, negative, anchor_class, negative_class in dataloader:
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            pos_sim = F.cosine_similarity(anchor_emb, positive_emb)
            neg_sim = F.cosine_similarity(anchor_emb, negative_emb)
            correct += (pos_sim > neg_sim).sum().item()
            total += anchor.size(0)
    print(f"Accuracy: {correct * 100 / total}")
