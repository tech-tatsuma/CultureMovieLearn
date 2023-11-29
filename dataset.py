import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

class VideoDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path = self.data_frame.iloc[idx, 0]
        label = self.data_frame.iloc[idx, 1]
        video, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')

        if self.transform:
            video = self.transform(video)

        label = torch.tensor(label, dtype=torch.float)
        
        return video, label