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
        # Replace the final classification layer with two linear layers for binary classification of each task
        self.current_head = nn.Linear(in_features, num_classes)
        self.next_head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Get features from the SlowFast backbone
        features = self.slowfast(x)
        # Define the forward pass for each task
        current_output = self.current_head(features)
        next_output = self.next_head(features)
        return current_output, next_output