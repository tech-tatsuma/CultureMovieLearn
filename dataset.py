import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import sys
import os

class VideoDataset(Dataset):
    def __init__(self, csv_file, transform=None, addpath=None, cache_dir='/data/furuya/graycache', usecache=False):
        # Get the DataFrame from the CSV file
        self.data_frame = pd.read_csv(csv_file)
        # Initialize the transform
        self.transform = transform
        self.add_path = addpath
        self.cache_dir = cache_dir
        self.usecache = usecache

        if usecache:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            self._cache_videos()

    def __len__(self):
        return len(self.data_frame)
    
    def _cache_videos(self):
        for idx in range(len(self.data_frame)):
            video_path = self.add_path + "/" + self.data_frame.iloc[idx, 0]
            cache_path = os.path.join(self.cache_dir, f'video_{idx}.pt')

            if not os.path.exists(cache_path):
                video, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')
                video = video.permute(3, 0, 1, 2)
                if self.transform:
                    video = self.transform(video)
                torch.save(video, cache_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.usecache:
            cache_path = os.path.join(self.cache_dir, f'video_{idx}.pt')
            video = torch.load(cache_path)

        else:
            video_path = self.add_path + "/" + self.data_frame.iloc[idx, 0]
            video, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')
            video = video.permute(3, 0, 1, 2)
            if self.transform:
                video = self.transform(video)

        current_label = self.data_frame.iloc[idx, 1]
        next_label = self.data_frame.iloc[idx + 1, 1] if idx + 1 < len(self.data_frame) else current_label

        if current_label == 2:
            current_label = 1
        if next_label == 2:
            next_label = 1

        current_label = torch.tensor(current_label, dtype=torch.long)
        next_label = torch.tensor(next_label, dtype=torch.long)

        return video, (current_label, next_label)