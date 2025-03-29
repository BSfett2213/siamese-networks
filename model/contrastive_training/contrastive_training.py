import time
import torch
import torchmetrics
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from model.model import SiameseNetwork
from skimage.filters import threshold_otsu
from torchvision.datasets import ImageFolder
from model.loss_types import ContrastiveLoss
from dataset.data_model import ContrastiveDataset

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

# Hyperparameters
BATCH_SIZE = 96
EPOCHS = 5
MARGIN = 0.2 ** 0.5

data = ImageFolder(root="../../dataset/extracted_faces", transform=transform)

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_subset, test_subset = torch.utils.data.random_split(data, [train_size, test_size])

train_dataset = ImageFolder(root="../../dataset/extracted_faces", transform=transform)
test_dataset = ImageFolder(root="../../dataset/extracted_faces", transform=transform)

train_dataset.samples = [data.samples[i] for i in train_subset.indices]
test_dataset.samples = [data.samples[i] for i in test_subset.indices]

train_dataset.targets = [label for _, label in train_dataset.samples]
test_dataset.targets = [label for _, label in test_dataset.samples]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss(margin=MARGIN).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

training_data = ContrastiveDataset(train_dataset, model, device)
testing_data = ContrastiveDataset(test_dataset, model, device)
train_loader = training_data.get_dataloader(BATCH_SIZE)
test_loader = testing_data.get_dataloader(BATCH_SIZE)

for epoch in range(EPOCHS):
    total_loss = 0
    epoch_start_time = time.time()

    for batch_idx, (img1, img2, label) in enumerate(train_loader, 1):
        print(f"\rBatch: {batch_idx}/{len(train_loader)}", end="")
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
    print(f"Epoch: {epoch + 1}/{EPOCHS}\nLoss: {total_loss / len(train_loader):.4f}")
    print(f"Time Taken: {epoch_time:.4f}s")
    print(f"Estimated Time to Finish: {remaining_time:.2f}s")

    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
    distances = []
    labels = []

    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            distance = F.pairwise_distance(output1, output2)
            distances.extend(distance.cpu().numpy())
            labels.extend(label.cpu().numpy())

    THRESHOLD = threshold_otsu(np.array(distances))
    labels = torch.tensor(np.array(labels), dtype=torch.float32, device=device).squeeze()
    predictions = torch.tensor(np.array([1 if d < THRESHOLD else 0 for d in distances]), dtype=torch.float32,
                               device=device)
    accuracy = accuracy_metric(predictions, labels)
    print(f"Model Accuracy: {accuracy.item() * 100:.2f}%")
    print("-" * 50)
