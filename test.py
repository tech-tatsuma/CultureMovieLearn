import torch

from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torchvision.io import read_video

import pandas as pd
import argparse
import datetime

from model import CustomSlowFast

def normalize_video(video, mean, std):
    normalized_video = []
    for frame in video.permute(1, 0, 2, 3):
        normalized_frame = Normalize(mean, std)(frame)
        normalized_video.append(normalized_frame)
    return torch.stack(normalized_video, dim=1)

def video2tensor(video_path):
    transform = Compose([
        UniformTemporalSubsample(25),
        Lambda(lambda x: x / 255.0),
        Lambda(lambda x: normalize_video(x, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])),
        Resize((256, 256)),
    ])

    video, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')
    video = video.permute(3, 0, 1, 2)
    video = transform(video)

    return video

def load_model(model_path, device):
    model = CustomSlowFast(device=device, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main(opt):
    csv_file = opt.csv_file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(opt.model_path, device)

    df = pd.read_csv(csv_file)
    predictions = []
    time_list = []

    for idx, row in df.iterrows():
        next_output = None
        if idx%2 == 0:
            video_path = row[0]
            start_time = datetime.datetime.now()
            video = video2tensor(video_path).to(device)
            video = video.unsqueeze(0)
            with torch.no_grad():
                current_output, next_output = model(video)
                predictions.append(current_output.item())
            end_time = datetime.datetime.now()
            time_list.append((end_time - start_time).total_seconds())
        else:
            predictions.append(next_output.item())

    df['prediction'] = predictions
    df.to_csv(opt.output_csv, index=False)
    average_time = sum(time_list)/len(time_list)
    print(f'Average time: {average_time} seconds')

if __name__ == '__main__':
    argumentparser = argparse.ArgumentParser()
    argumentparser.add_argument('--csv_file', type=str, required=True)
    argumentparser.add_argument('--model_path', type=str, required=True)
    argumentparser.add_argument('--output_csv', type=str, required=True)
    opt = argumentparser.parse_args()
    main(opt)
    