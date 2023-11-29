import torch
import torch.nn as nn
import torch.hub

# Custom neural network class using the SlowFast architecture for video classification
class CustomSlowFast(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomSlowFast, self).__init__()
        # Load a pretrained SlowFast model from PyTorch Hub
        self.slowfast = torch.hub.load(
            'facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True
        )
        # Get the number of input features for the final classification layer
        in_features = self.slowfast.head.projection.in_features
        # Replace the final classification layer with a linear layer for binary classification
        self.slowfast.head.projection = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Define the forward pass through the network
        return self.slowfast(x)