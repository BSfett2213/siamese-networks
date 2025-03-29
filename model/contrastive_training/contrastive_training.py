import numpy as np
import torch
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from model.model import SiameseNetwork
from torchvision.datasets import ImageFolder
from model.loss_types import ContrastiveLoss
from dataset.data_model import ContrastiveDataset
from skimage.filters import threshold_otsu

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
MARGIN = 0.5

data = ImageFolder(root="../../dataset/extracted_faces", transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiameseNetwork().to(device)
criterion = ContrastiveLoss(margin=MARGIN).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

contrastive_data = ContrastiveDataset(data, model, device)
contrastive_loader = contrastive_data.get_dataloader(BATCH_SIZE)

for epoch in range(EPOCHS):
    total_loss = 0
    epoch_start_time = time.time()

    for batch_idx, (img1, img2, label) in enumerate(contrastive_loader, 1):
        print(f"\rBatch: {batch_idx}/{len(contrastive_loader)}", end="")
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_time = time.time() - epoch_start_time
    remaining_time = epoch_time * (EPOCHS - (epoch + 1))
    print("\n" + "-" * 50)
    print(f"Epoch: {epoch + 1}/{EPOCHS}\nLoss: {total_loss / len(contrastive_loader):.4f}")
    print(f"Time Taken: {epoch_time:.4f}s")
    print(f"Estimated Time to Finish: {remaining_time:.2f}s")
    print("-" * 50)

model.eval()
correct = 0
total = 0
distances = []
labels = []

with torch.no_grad():
    for img1, img2, label in contrastive_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        output1, output2 = model(img1, img2)
        distances.extend(F.pairwise_distance(output1, output2).cpu().numpy())
        labels.extend(label.squeeze().cpu().numpy())

plt.hist(distances, bins=50, alpha=0.6, label="Distances")
plt.legend(), plt.xlabel("Distance"), plt.ylabel("Frequency")
plt.savefig("distances.svg", format="svg")

accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
distances = []
labels = []

with torch.no_grad():
    for img1, img2, label in contrastive_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        output1, output2 = model(img1, img2)
        distance = F.pairwise_distance(output1, output2)
        distances.extend(distance.cpu().numpy())
        labels.extend(label.cpu().numpy())

THRESHOLD = threshold_otsu(np.array(distances))
labels = torch.tensor(np.array(labels), dtype=torch.float32, device=device).squeeze()
predictions = torch.tensor(np.array([1 if d < THRESHOLD else 0 for d in distances]), dtype=torch.float32, device=device)
accuracy = accuracy_metric(predictions, labels)
print(f"Model Accuracy: {accuracy.item() * 100:.2f}%")
