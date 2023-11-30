import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

class VideoDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Get the DataFrame from the CSV file
        self.data_frame = pd.read_csv(csv_file)
        # Initialize the transform
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the video path and label from the DataFrame
        video_path = self.data_frame.iloc[idx, 0]
        # Get the current and next label
        current_label = self.data_frame.iloc[idx, 1]
        # Check if next index exists, if not use current label again
        if idx + 1 < len(self.data_frame):
            next_label = self.data_frame.iloc[idx + 1, 1]
        else:
            next_label = current_label
        # Read the video from the path
        video, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')
        # Apply the transform to the video
        if self.transform:
            video = self.transform(video)
        # Change the labels to torch tensors
        current_label = torch.tensor(current_label, dtype=torch.float)
        next_label = torch.tensor(next_label, dtype=torch.float)
        
        return video, (current_label, next_label)