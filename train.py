import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.transforms import functional as F

from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda, Normalize, Resize, Grayscale

import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import sys
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")

from model import CustomSlowFast
from dataset import VideoDataset

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
            frame = F.adjust_brightness(frame, brightness_factor)
            frame = F.adjust_contrast(frame, contrast_factor)
            frame = F.adjust_saturation(frame, saturation_factor)
            frame = F.adjust_hue(frame, hue_factor)
            transformed_video.append(frame)

        return torch.stack(transformed_video).permute(1, 0, 2, 3)

def normalize_video(video, mean, std):
    normalized_video = []
    for frame in video.permute(1, 0, 2, 3):
        normalized_frame = Normalize(mean, std)(frame)
        normalized_video.append(normalized_frame)
    return torch.stack(normalized_video, dim=1)


# Function for setting seeds for reproducibility
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to calculate the size of the model
def calculate_model_size(model):
    total_size = 0
    for param in model.parameters():
        param_size = param.numel()
        total_size += param_size * param.element_size()

    total_size_bytes = total_size
    total_size_kb = total_size / 1024
    total_size_mb = total_size_kb / 1024
    total_size_gb = total_size_mb / 1024

    print(f"Model size: {total_size_bytes} bytes / {total_size_kb:.2f} KB / {total_size_mb:.2f} MB / {total_size_gb:.4f} GB")

class VideoGrayscale:
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, video):
        grayscale_frames = []
        for t in range(video.size(1)):
            frame = video[:, t, :, :]
            frame = F.to_pil_image(frame) 
            frame = F.to_grayscale(frame, num_output_channels=self.num_output_channels)
            frame = F.to_tensor(frame)
            grayscale_frames.append(frame)

        return torch.stack(grayscale_frames, dim=1) 


# Main training function
def train(opt):

    # Setting the seed for reproducibility
    seed_everything(opt.seed)

    # Setting up the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extracting values from command line options
    csv_file = opt.data
    epochs = opt.epochs
    patience = opt.patience
    learning_rate = opt.lr
    batch = opt.batch

    addpath = os.path.dirname(csv_file)

    # Data preprocessing steps
    transform = Compose([
        UniformTemporalSubsample(25),
        Lambda(lambda x: x / 255.0),
        VideoGrayscale(num_output_channels=3),
        Lambda(lambda x: normalize_video(x, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])),
        Resize((256, 256)),
        # VideoColorJitter(brightness=0.5)
    ])

    # Creating the dataset
    video_dataset = VideoDataset(csv_file=csv_file, transform=transform, addpath=addpath)
    # Splitting the data into training, validation, and test sets
    train_size = int(0.7 * len(video_dataset))
    val_size = len(video_dataset) - train_size
    train_dataset, val_dataset = random_split(video_dataset, [train_size, val_size])
    # Creating data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # Initializing the model
    model = CustomSlowFast(device=device, num_classes=1)

    # Utilizing multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Calculating model size
    calculate_model_size(model)

    # Setting up the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = BCEWithLogitsLoss()

    # Initialize minimum validation loss for early stopping
    val_loss_min = None
    val_loss_min_epoch = 0

    # Arrays to record the training and validation results
    train_losses = []
    val_losses = []
    val_accuracy = []

    # Training loop
    for epoch in tqdm(range(epochs)):
        # Set model to training mode
        model.train()
        # Reset epoch's training and validation loss
        train_loss = 0.0
        val_loss = 0.0

        # Training phase
        for i, (inputs, (current_labels, next_labels)) in enumerate(train_loader):
            inputs = inputs.to(device)
            current_labels = current_labels.to(device)
            next_labels = next_labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass and backward pass
            current_outputs, next_outputs = model(inputs)
            loss_current = criterion(current_outputs, current_labels.unsqueeze(1))
            loss_next = criterion(next_outputs, next_labels.unsqueeze(1))
            # Combine losses for multitasking
            loss = loss_current + loss_next
            loss.backward()
            optimizer.step()
            # Accumulate the loss
            train_loss += loss.item()

        # Calculate loss per epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Set model to evaluation mode
        model.eval()

        # Initialize variables for accuracy calculation
        correct = 0
        total = 0

        # Validation phase
        with torch.no_grad():
            for i, (inputs, (current_labels, next_labels)) in enumerate(val_loader):
                inputs = inputs.to(device)
                current_labels = current_labels.to(device)
                next_labels = next_labels.to(device)
                # Forward pass
                current_outputs, next_outputs = model(inputs)
                # Calculate predictions
                probs_current = torch.sigmoid(current_outputs)
                probs_next = torch.sigmoid(next_outputs)
                predicted_current = probs_current >= 0.5
                predicted_next = probs_next >= 0.5
                # Accumulate validation losses for both tasks
                val_loss += criterion(current_outputs, current_labels).item()
                val_loss += criterion(next_outputs, next_labels).item()
                # Accumulate test results for both tasks
                total += current_labels.size(0)
                correct += (predicted_current == current_labels).sum().item()
                total += next_labels.size(0)
                correct += (predicted_next == next_labels).sum().item()
                
        # Calculate validation accuracy and loss
        accuracy = 100 * correct / total
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy.append(accuracy)

        print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
        sys.stdout.flush()

        # Check for early stopping
        if val_loss_min is None or val_loss < val_loss_min:
            # Save model if validation loss decreased
            model_save_name = f'./latestresult/lr{learning_rate}_ep{epochs}_pa{patience}.pt'
            torch.save(model.state_dict(), model_save_name)
            val_loss_min = val_loss
            val_loss_min_epoch = epoch
        elif (epoch - val_loss_min_epoch) >= patience:
            # Early stopping if no improvement in validation loss
            print('Early stopping due to validation loss not improving for {} epochs'.format(patience))
            sys.stdout.flush()
            break

    # Plot training and validation loss and accuracy
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.title("Training and Validation CrossEntropy Loss")
    plt.title("Accuracy")
    plt.savefig(f'./latestgraph/lr{learning_rate}_ep{epochs}_pa{patience}.png')

    # Return final training and validation loss
    return train_loss, val_loss_min

# Main block to execute the script
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print('start time:',start_time)
    sys.stdout.flush()

    # Argument parser for command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/labels.csv', help='path to data file')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr_search', type=str, default='false', help='whether to perform learning rate search')
    opt = parser.parse_args()

    # Learning rate search if specified
    if opt.lr_search == 'true':
        # Perform learning rate search
        learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
        best_loss = float('inf')
        best_lr = 0
        train_losses = []
        val_losses = []
        for lr in learning_rates:
            opt.lr = lr
            print(f"\nTraining with learning rate: {lr}")
            sys.stdout.flush()
            print('-----beginning training-----')
            sys.stdout.flush()

            train_loss, val_loss = train(opt)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_lr = lr
        print('best validation loss: ', best_loss)
        sys.stdout.flush()
        print(f"Best learning rate: {best_lr}")
        sys.stdout.flush()
        print('Learning rate search results:')
        for lr, train_loss, val_loss in zip(learning_rates, train_losses, val_losses):
            print(f'Learning rate: {lr}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')
    else:
        # Train the model with the given hyperparameters.
        print('-----biginning training-----')
        sys.stdout.flush()
        train_loss, val_loss = train(opt)
        print('final train loss: ',train_loss)
        print('final validation loss: ', val_loss)
        sys.stdout.flush()

    # Print script execution time
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print('-----completing training-----')
    sys.stdout.flush()
    print('end time:',end_time)
    sys.stdout.flush()
    print('Execution time: ', execution_time)
    sys.stdout.flush()