"""Classification output head on the shared image Transformer decoder."""
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, findings=13, width=256):
        super().__init__()
        self.output = nn.Linear(width, findings)

    def forward(self, image_features):
        return self.output(image_features.mean(1))
