import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="whitegrid")
palette = sns.color_palette("husl", 3)
triplet_csv = pd.read_csv("./triplet_training/triplet_results.csv")
triplet_cosine_csv = pd.read_csv("./triplet_cosine_training/training_results.csv")
contrastive_csv = pd.read_csv("./contrastive_training/contrastive_results.csv")

plt.figure(figsize=(10, 6))
sns.lineplot(x=triplet_csv['Epoch'], y=triplet_csv['Loss'], color=palette[0], label="Triplet Loss", linewidth=2.5,
             marker="o")
sns.lineplot(x=triplet_csv['Epoch'], y=triplet_cosine_csv['Loss'], color=palette[1], label="Triplet Cosine Loss",
             linewidth=2.5, marker="o")
sns.lineplot(x=triplet_csv['Epoch'], y=contrastive_csv['Loss'], color=palette[2], label="Contrastive Loss",
             linewidth=2.5, marker="o")
plt.ylim(0, 1.0)
plt.title("Training Loss per Epoch", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12), plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.savefig("loss_comparison.svg", format='svg')
plt.close()

plt.figure(figsize=(10, 6))
sns.lineplot(x=triplet_csv['Epoch'], y=triplet_csv['Accuracy'], color=palette[0], label="Triplet Accuracy", linewidth=2.5,
             marker="o")
sns.lineplot(x=triplet_csv['Epoch'], y=triplet_cosine_csv['Accuracy'], color=palette[1], label="Triplet Cosine Accuracy",
             linewidth=2.5, marker="o")
sns.lineplot(x=triplet_csv['Epoch'], y=contrastive_csv['Accuracy'], color=palette[2], label="Contrastive Accuracy",
             linewidth=2.5, marker="o")
plt.ylim(45, 80)
plt.title("Training Accuracy per Epoch", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12), plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.savefig("accuracy_comparison.svg", format='svg')
plt.close()
