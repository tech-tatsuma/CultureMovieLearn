import torch
import torch.nn.functional as F

from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torchvision.io import read_video
from torchvision.transforms import functional as FT

import pandas as pd
import argparse
import datetime
import os
import sys
import matplotlib.pyplot as plt
import random

from model import CustomSlowFast

# Class for applying random crops to video brightness
class VideoColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, video):
        video = video.permute(1, 0, 2, 3)
        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        transformed_video = []
        for frame in video:
            frame = FT.adjust_brightness(frame, brightness_factor)
            frame = FT.adjust_contrast(frame, contrast_factor)
            frame = FT.adjust_saturation(frame, saturation_factor)
            frame = FT.adjust_hue(frame, hue_factor)
            transformed_video.append(frame)

        return torch.stack(transformed_video).permute(1, 0, 2, 3)

class VideoGrayscale:
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, video):
        grayscale_frames = []
        for t in range(video.size(1)):
            frame = video[:, t, :, :]
            frame = FT.to_pil_image(frame) 
            frame = FT.to_grayscale(frame, num_output_channels=self.num_output_channels)
            frame = FT.to_tensor(frame)
            grayscale_frames.append(frame)

        return torch.stack(grayscale_frames, dim=1) 

def normalize_video(video, mean, std):
    normalized_video = []
    for frame in video.permute(1, 0, 2, 3):
        normalized_frame = Normalize(mean, std)(frame)
        normalized_video.append(normalized_frame)
    return torch.stack(normalized_video, dim=1)

def video2tensor(video_path, is_color=True):
    if is_color:
        transform = Compose([
            UniformTemporalSubsample(25),
            Lambda(lambda x: x / 255.0),
            Lambda(lambda x: normalize_video(x, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])),
            Resize((256, 256)),
            VideoColorJitter(brightness=0.5)
        ])
    else:
        transform = Compose([
            UniformTemporalSubsample(25),
            Lambda(lambda x: x / 255.0),
            VideoGrayscale(num_output_channels=3),
            Lambda(lambda x: normalize_video(x, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])),
            Resize((256, 256)),
        ])

    video, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')
    video = video.permute(3, 0, 1, 2)
    video = transform(video)

    return video

def load_model(model_path, device):
    model = CustomSlowFast(device=device, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main(opt):
    csv_file = opt.csv_file
    addpath = os.path.dirname(csv_file)
    device_type = opt.device
    color = opt.color
    if color == 'true':
        color_bool = True
    else:
        color_bool = False

    if device_type == 'GPU':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = load_model(opt.model_path, device)

    df = pd.read_csv(csv_file)
    predictions = []
    time_list = []
    true_values = []

    for idx, row in df.iterrows():
        predicted_next = None
        if idx%2 == 0:
            video_path = row[0]
            video_path = os.path.join(addpath, video_path)
            start_time = datetime.datetime.now()
            video = video2tensor(video_path, color_bool).to(device)
            video = video.unsqueeze(0)
            with torch.no_grad():
                current_outputs, next_outputs = model(video)
                print(f"current_output: {current_outputs}")
                print(f"next_output: {next_outputs}")  
                sys.stdout.flush()

                # Predicted class with highest probability
                _, predicted_current = torch.max(current_outputs, 1)
                _, predicted_next = torch.max(next_outputs, 1)
                predictions.append(predicted_current.item())
                predictions.append(predicted_next.item())

            end_time = datetime.datetime.now()
            time_list.append((end_time - start_time).total_seconds())

        true_value = 1 if row['status'] == 2 else row['status']
        true_values.append(true_value)

    df['prediction'] = predictions
    df.to_csv(opt.output_csv, index=False)
    average_time = sum(time_list)/len(time_list)
    print(f'Average time: {average_time} seconds')

    plt.figure()
    plt.scatter(range(len(predictions)), predictions, c='blue', label='pred', marker='o')

    change_points = [i for i in range(1, len(true_values)) if true_values[i] != true_values[i-1]]
    for i in change_points:
        plt.plot([i-1, i], true_values[i-1:i+1], c='orange', linestyle='-', marker='x')

    plt.title("Test Predictions")
    plt.xlabel("Time")
    plt.ylabel("Prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig('./test_predictions.png')

if __name__ == '__main__':
    argumentparser = argparse.ArgumentParser()
    argumentparser.add_argument('--csv_file', type=str, required=True)
    argumentparser.add_argument('--model_path', type=str, required=True)
    argumentparser.add_argument('--output_csv', type=str, required=True)
    argumentparser.add_argument('--color', type=str, default='false')
    argumentparser.add_argument('--device', type=str, default='GPU')
    opt = argumentparser.parse_args()
    print(opt)
    main(opt)
    