import torch

from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torchvision.io import read_video

import pandas as pd
import argparse
import datetime
import os
import sys
import matplotlib.pyplot as plt

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
    addpath = os.path.dirname(csv_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(opt.model_path, device)

    df = pd.read_csv(csv_file)
    predictions = []
    time_list = []
    plot_list = []
    prediction_numbers = []

    for idx, row in df.iterrows():
        predicted_next = None
        if idx%2 == 0:
            video_path = row[0]
            video_path = os.path.join(addpath, video_path)
            start_time = datetime.datetime.now()
            video = video2tensor(video_path).to(device)
            video = video.unsqueeze(0)
            with torch.no_grad():
                current_output, next_output_origin = model(video)
                plot_list.append(current_output.item())
                plot_list.append(next_output_origin.item())
                probs_current = torch.sigmoid(current_output)
                probs_next = torch.sigmoid(next_output_origin)
                predicted_current = (probs_current >= 0.5).squeeze()
                predicted_next = (probs_next >= 0.5).squeeze()
                predictions.append(predicted_current.item())
                predicted_next = predicted_next.item()
                predictions.append(predicted_next)
                prediction_numbers.append(current_output.item())
                prediction_numbers.append(next_output_origin.item())
            end_time = datetime.datetime.now()
            time_list.append((end_time - start_time).total_seconds())

    df['prediction'] = predictions
    df['prediction_number'] = prediction_numbers
    df.to_csv(opt.output_csv, index=False)
    average_time = sum(time_list)/len(time_list)
    print(f'Average time: {average_time} seconds')

    plt.plot(plot_list, marker='o', linestyle='-')
    plt.title("Test Predictions")
    plt.xlabel("Time")
    plt.ylabel("Prediction")
    plt.grid(True)
    plt.savefig('./test_predictions.png')

if __name__ == '__main__':
    argumentparser = argparse.ArgumentParser()
    argumentparser.add_argument('--csv_file', type=str, required=True)
    argumentparser.add_argument('--model_path', type=str, required=True)
    argumentparser.add_argument('--output_csv', type=str, required=True)
    opt = argumentparser.parse_args()
    main(opt)
    