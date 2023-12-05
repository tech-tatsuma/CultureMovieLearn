import torch
import torch.nn as nn
import torch.hub
import torch.nn.functional as F

class CustomAvgPool3d(nn.Module):
    def __init__(self):
        super(CustomAvgPool3d, self).__init__()
    # Define the kernel size for average pooling based on the input dimensions.
    # For the second and third dimensions of the kernel, it takes the minimum of the input size and 7.
    def forward(self, x):
        kernel_size = (x.size(2), min(x.size(3), 7), min(x.size(4), 7))
        return F.avg_pool3d(x, kernel_size=kernel_size, stride=(1, 1, 1), padding=(0, 0, 0))
    
# Custom neural network class using the SlowFast architecture for video classification
class CustomSlowFast(nn.Module):
    def __init__(self, device, num_classes=2):
        super(CustomSlowFast, self).__init__()
        self.device = device
        # Load the SlowFast architecture from PyTorch Hub, not pretrained
        self.slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
        in_features = 400
        # Replace the average pooling layers in the SlowFast blocks with custom average pooling
        self.slowfast.blocks[5].pool[0] = CustomAvgPool3d()
        self.slowfast.blocks[5].pool[1] = CustomAvgPool3d()
        # Linear layers for classification tasks
        self.current_head = nn.Linear(in_features, num_classes)
        self.next_head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Number of frames in the input video
        num_frames = x.shape[2]

        # Slow pathway sampling rate
        slow_rate = 4
        # Calculate indices for the slow pathway frames
        slow_frames = torch.linspace(0, num_frames - 1, num_frames // slow_rate).long().to(self.device)
        # Extract frames for the slow pathway
        slow_pathway = x[:, :, slow_frames]
        
        # Fast pathway uses all frames
        fast_pathway = x

        # If the number of frames is not divisible by the slow rate, 
        # trim the fast pathway to match the slow pathway length
        if slow_pathway.size(2) != fast_pathway.size(2):
            fast_pathway = fast_pathway[:, :, :slow_pathway.size(2) * slow_rate]

        # Combine slow and fast pathways
        frame_list = [slow_pathway, fast_pathway]
        # Pass through the SlowFast network
        features = self.slowfast(frame_list)

        # Apply linear layers to the output features for classification
        current_output = self.current_head(features)
        next_output = self.next_head(features)

        return current_output, next_output
