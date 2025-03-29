import torch
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from model.model import SiameseNetwork
from torchvision.datasets import ImageFolder
from model.loss_types import ContrastiveLoss
from dataset.data_model import ContrastiveDataset

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

BATCH_SIZE = 64
EPOCHS = 15

data = ImageFolder(root="../../dataset/extracted_faces", transform=transform)
contrastive_data = ContrastiveDataset(data)
contrastive_loader = contrastive_data.get_dataloader(BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiameseNetwork().to(device)
criterion = ContrastiveLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    total_loss = 0
    epoch_start_time = time.time()

    for batch_idx, (img1, img2, label) in enumerate(contrastive_loader, 1):
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
    print(f"Epoch: {epoch + 1}/{EPOCHS}\nLoss: {total_loss/len(contrastive_loader):.4f}")
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


with torch.no_grad():
    for img1, img2, label in contrastive_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        output1, output2 = model(img1, img2)
        distance = F.pairwise_distance(output1, output2)

        predictions = (distance < 0.1).float()
        label = label.squeeze()

        correct += (predictions == label).sum().item()
        total += label.numel()

accuracy = 100 * correct / total
print(f"Model Accuracy: {accuracy:.2f}%")
