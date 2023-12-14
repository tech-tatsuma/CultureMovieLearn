import torch

from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torchvision.io import read_video

from fastapi import FastAPI, File, UploadFile
import uvicorn
from typing import Dict
import shutil
import os

from model import CustomSlowFast

def load_model(model_path, device):
    model = CustomSlowFast(device=device, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

app = FastAPI()

# replace this with your model path
model_path = "path_to_model_file"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(model_path, device)

def normalize_video(video, mean, std):
    normalized_video = []
    for frame in video.permute(1, 0, 2, 3):
        normalized_frame = Normalize(mean, std)(frame)
        normalized_video.append(normalized_frame)
    return torch.stack(normalized_video, dim=1)

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

def video2tensor(video_path):
    transform = Compose([
        UniformTemporalSubsample(25),
        Lambda(lambda x: x / 255.0),
        Lambda(lambda x: normalize_video(x, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])),
        Resize((256, 256)),
        VideoColorJitter(brightness=0.3)
    ])

    video, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')
    video = video.permute(3, 0, 1, 2)
    video = transform(video)

    return video

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_tensor = video2tensor(temp_file_path)

    with torch.no_grad():
        current_output, next_output = model(video_tensor.unsqueeze(0))
        
        predicted_current = torch.sigmoid(current_outputs)
        predicted_next = torch.sigmoid(next_outputs)

    os.remove(temp_file_path)

    return {"current": predicted_current.item(), "next": predicted_next.item()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)