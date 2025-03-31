import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set seaborn style
sns.set_theme(style="whitegrid")
palette = sns.color_palette("husl", 2)
contrastive_csv = pd.read_csv("../../model/contrastive_training/contrastive_results.csv")

# Plotting Loss
plt.figure(figsize=(10, 6))
sns.lineplot(x=contrastive_csv['Epoch'], y=contrastive_csv['Loss'], color=palette[0], label="Loss", linewidth=2.5,
             marker="o")
plt.title("Contrastive Training Loss per Epoch", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.savefig("contrastive_loss.svg", format='svg')
plt.close()

# Plotting Accuracy
plt.figure(figsize=(10, 6))
sns.lineplot(x=contrastive_csv['Epoch'], y=contrastive_csv['Accuracy'], color=palette[1], label="Accuracy",
             linewidth=2.5, marker="o")
plt.title("Contrastive Training Accuracy per Epoch", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.savefig("contrastive_accuracy.svg", format='svg')
plt.close()
