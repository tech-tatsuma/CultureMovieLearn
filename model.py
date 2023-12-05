import torch
import torch.nn as nn
import torch.hub
import torch.nn.functional as F

class CustomAvgPool3d(nn.Module):
    def __init__(self):
        super(CustomAvgPool3d, self).__init__()

    def forward(self, x):
        kernel_size = (x.size(2), min(x.size(3), 7), min(x.size(4), 7))
        return F.avg_pool3d(x, kernel_size=kernel_size, stride=(1, 1, 1), padding=(0, 0, 0))
    
# Custom neural network class using the SlowFast architecture for video classification
class CustomSlowFast(nn.Module):
    def __init__(self, device, num_classes=2):
        super(CustomSlowFast, self).__init__()
        self.device = device
        self.slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
        in_features = 400
        self.slowfast.blocks[5].pool[0] = CustomAvgPool3d()
        self.slowfast.blocks[5].pool[1] = CustomAvgPool3d()
        self.current_head = nn.Linear(in_features, num_classes)
        self.next_head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # num of frames
        num_frames = x.shape[2]

        # sampling rate
        slow_rate = 4
        slow_frames = torch.linspace(0, num_frames - 1, num_frames // slow_rate).long().to(self.device)
        slow_pathway = x[:, :, slow_frames]
        
        # fast pathway
        fast_pathway = x

        # if the number of frames is not divisible by the slow rate, then we need to cut off some frames from the fast pathway
        if slow_pathway.size(2) != fast_pathway.size(2):
            fast_pathway = fast_pathway[:, :, :slow_pathway.size(2) * slow_rate]

        frame_list = [slow_pathway, fast_pathway]
        features = self.slowfast(frame_list)

        current_output = self.current_head(features)
        next_output = self.next_head(features)

        return current_output, next_output
