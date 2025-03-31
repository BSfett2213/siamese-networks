import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set seaborn style
sns.set_theme(style="whitegrid")
palette = sns.color_palette("husl", 2)
triplet_csv = pd.read_csv("triplet_results.csv")

# Plotting Triplet Loss
plt.figure(figsize=(10, 6))
sns.lineplot(x=triplet_csv['Epoch'], y=triplet_csv['Loss'], color=palette[0], label="Loss", linewidth=2.5, marker="o")
plt.title("Triplet Training Loss per Epoch", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12), plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.savefig("triplet_loss.svg", format='svg')
plt.close()

# Plotting Triplet Accuracy
plt.figure(figsize=(10, 6))
sns.lineplot(x=triplet_csv['Epoch'], y=triplet_csv['Accuracy'], color=palette[1], label="Accuracy", linewidth=2.5,
             marker="o")
plt.title("Triplet Training Accuracy per Epoch", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12), plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.savefig("triplet_accuracy.svg", format='svg')
plt.close()
