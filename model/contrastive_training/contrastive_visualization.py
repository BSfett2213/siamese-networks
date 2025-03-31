import matplotlib.pyplot as plt; import numpy as np
import pandas as pd
#Loading results csv
triplet_csv = pd.read_csv("../../model/triplet_training/triplet_results.csv")
contrastive_csv = pd.read_csv("../../model/contrastive_training/contrastive_results.csv")


# Plotting loss
plt.figure(figsize = (10,10))
plt.title("Contrastive training Loss per Epoch")
y_loss_contrastive = np.array(contrastive_csv['Loss'])
plt.plot(y_loss_contrastive, color = 'r', label = "Loss", linewidth = 2)
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.xlim(0, contrastive_csv.Epoch.max())
plt.legend()
plt.savefig("../../model/contrastive_training/contrastive_loss.svg", format='svg')



# Plotting Accuracy
plt.title("Contrastive Training Accuracy per Epoch")
y_acc_contrastive = np.array(contrastive_csv['Accuracy'])
plt.plot(y_acc_contrastive, color = 'y', label = "Accuracy", linewidth = 2)
plt.xlabel('Epochs'), plt.ylabel('Accuracy')
plt.xlim(0, contrastive_csv.Epoch.max())
plt.legend()
plt.savefig("../../model/contrastive_training/contrastive_accuracy.svg", format='svg')


# Plotting Contrastive Training Accuracy vs Triplet Training Accuracy
plt.title("Contrastive Training vs Triplet Training Accuracy per Epoch")
y_acc_triplet = np.array(triplet_csv['Accuracy'])
plt.plot(y_acc_contrastive, color = 'y', label = "Accuracy", linewidth = 2)
plt.plot(y_acc_triplet, color = 'black', label = "Accuracy", linewidth = 2)
plt.xlabel('Epochs'), plt.ylabel('Accuracy')
plt.xlim(0, triplet_csv.Epoch.max())
plt.legend()
plt.savefig("../../model/contrastive_training/comparison.svg", format='svg')
plt.show()


# Debugging/// DO NOT RUN YET