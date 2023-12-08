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
model_path = "path_to_model_file"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(model_path, device)

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

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_tensor = video2tensor(temp_file_path)

    with torch.no_grad():
        current_output, next_output = model(video_tensor.unsqueeze(0))
        
        _, predicted_current = torch.max(current_outputs, 1)
        _, predicted_next = torch.max(next_outputs, 1)

    os.remove(temp_file_path)

    return {"current": predicted_current.item(), "next": predicted_next.item()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)