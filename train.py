import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from pytorchvideo.transforms import UniformTemporalSubsample, ColorJitter
from torchvision.transforms import Compose, Lambda, Normalize, Resize

import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import sys

from .model import CustomSlowFast
from .dataset import VideoDataset

# Class for applying random crops to video brightness
class VideoColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, video):
        # Initialize the transformation for consistent application across the whole video
        transform = self.color_jitter.get_params(self.color_jitter.brightness, self.color_jitter.contrast,
                                                 self.color_jitter.saturation, self.color_jitter.hue)

        # Apply the same transformation to each frame
        return torch.stack([transform(frame) for frame in video])

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

    # Data preprocessing steps
    transform = Compose([
        UniformTemporalSubsample(8),
        Lambda(lambda x: x / 255.0),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        Resize((256, 256)),
        VideoColorJitter(brightness=0.5)
    ])

    # Creating the dataset
    video_dataset = VideoDataset(csv_file=csv_file, transform=transform)
    # Splitting the data into training, validation, and test sets
    train_size = int(0.7 * len(video_dataset))
    val_test_size = len(video_dataset) - train_size
    train_dataset, val_test_dataset = random_split(video_dataset, [train_size, val_test_size])
    test_size = int(0.33 * len(val_test_dataset))
    val_dataset, test_dataset = random_split(val_test_dataset, [test_size, len(val_test_dataset) - test_size])
    # Creating data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # Initializing the model
    model = CustomSlowFast(num_classes=1)

    # Utilizing multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Calculating model size
    calculate_model_size(model)

    # Setting up the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion = BCEWithLogitsLoss()

    # Initialize minimum validation loss for early stopping
    val_loss_min = None
    val_loss_min_epoch = 0

    # Arrays to record the training and validation results
    train_losses = []
    val_losses = []
    val_accuracy = []

    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        # Reset epoch's training and validation loss
        train_loss = 0.0
        val_loss = 0.0

        # Training phase
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass and backward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(inputs)
                # Calculate predictions
                probs = torch.sigmoid(outputs)
                predicted = probs >= 0.5
                # Accumulate test results
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # Accumulate validation loss
                val_loss += criterion(outputs, labels).item()
                
        # Calculate validation accuracy and loss
        accuracy = 100 * correct / total
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy.append(accuracy)

        print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

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
            break

    # Test phase
    test_loss = []
    test_correct = 0
    test_total = 0  

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs
            labels = labels.to(device)
            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predicted = probs >= 0.5
            # Accumulate test results
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            cross_loss += criterion(outputs, labels).item()
            test_loss.append(cross_loss)

    # Calculate test loss and accuracy
    mean_test_loss = sum(test_loss) / len(test_loss)
    test_accuracy = 100 * test_correct / test_total
    print(f'Test MSE: {mean_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

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