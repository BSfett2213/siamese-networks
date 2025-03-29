import torch
import time
import torch.optim as optim
import torch.nn.functional as F
from model.model import SiameseNetwork
from dataset.data_model import TripletDataset
from model.loss_types import TripletLoss
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
MARGIN = 1.0

data = ImageFolder(root="../../dataset/extracted_faces", transform=transform)
triplet_data = TripletDataset(data)
triplet_loader = triplet_data.get_dataloader(BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiameseNetwork().to(device)
criterion = TripletLoss(margin=MARGIN).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    total_loss = 0
    epoch_start_time = time.time()

    for batch_idx, (anchor, positive, negative) in enumerate(triplet_loader, 1):
        print(f"\rBatch: {batch_idx}/{len(triplet_loader)}", end="")
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        anchor_out, positive_out = model(anchor, positive)
        _, negative_out = model(anchor, negative)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_time = time.time() - epoch_start_time
    remaining_time = epoch_time * (EPOCHS - (epoch + 1))

    print("\n" + "-" * 50)
    print(f"Epoch: {epoch + 1}/{EPOCHS}\nLoss: {total_loss / len(triplet_loader):.4f}")
    print(f"Time Taken: {epoch_time:.4f}s")
    print(f"Estimated Time to Finish: {remaining_time:.2f}s")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for anchor, positive, negative in triplet_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_out, positive_out = model(anchor, positive)
            _, negative_out = model(anchor, negative)

            pos_dist = F.pairwise_distance(anchor_out, positive_out)
            neg_dist = F.pairwise_distance(anchor_out, negative_out)

            correct += (pos_dist < neg_dist).sum().item()
            total += anchor.size(0)

    accuracy = 100 * correct / total
    print(f"Model Accuracy: {accuracy:.2f}%")
    print("-" * 50)
