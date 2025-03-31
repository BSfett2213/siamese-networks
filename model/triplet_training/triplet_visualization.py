import matplotlib.pyplot as plt; import numpy as np; import pandas as pd
#Loading results csv
triplet_csv = pd.read_csv("../../model/triplet_training/triplet_results.csv")
contrastive_csv = pd.read_csv("../../model/contrastive_training/contrastive_results.csv")
# Plotting loss
plt.figure(figsize = (10, 10))
plt.subplot(2, 2, 1)
plt.title("Triplet training Loss per Epoch")
y_loss = np.array(triplet_csv['Loss'])
plt.plot(y_loss, color = 'r', label = "Loss", linewidth = 2)
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.xlim(0, triplet_csv.Epoch.max())
plt.legend()
# Plotting Accuracy
plt.subplot(2, 2, 2)
plt.title("Triplet Training Accuracy per Epoch")
y_acc = np.array(triplet_csv['Accuracy'])
plt.plot(y_acc, color = 'y', label = "Accuracy", linewidth = 2)
plt.xlabel('Epochs'), plt.ylabel('Accuracy')
plt.xlim(0, triplet_csv.Epoch.max())
plt.legend()
plt.subplot(2, 2, 3)
plt.title("Triplet Training Accuracy vs Contrastive Training Loss per Epoch")
y_acc_contrastive = np.array(contrastive_csv['Accuracy'])
plt.plot(y_acc, color = 'y', label = "Accuracy", linewidth = 2)
plt.plot(y_acc_contrastive, color = 'black', label = "Accuracy", linewidth = 2)
plt.xlabel('Epochs'), plt.ylabel('Accuracy')
plt.xlim(0, triplet_csv.Epoch.max())
plt.legend()
plt.show()


