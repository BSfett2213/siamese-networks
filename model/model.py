import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(36864, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward_image(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, image1, image2):
        output1 = self.forward_image(image1)
        output2 = self.forward_image(image2)
        return output1, output2
