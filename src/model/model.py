import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(4, 2)),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(4, 2)),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5)),
            nn.ReLU()
        )
        self.l4 = nn.AdaptiveAvgPool2d((1, 1))
        self.l5 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.l6 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.l5(x)
        x = self.l6(x)
        return x