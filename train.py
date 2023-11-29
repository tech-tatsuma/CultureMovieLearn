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

# 動画の明るさのランダムクロップのためのクラス
class VideoColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, video):
        # 動画全体に対して一貫した変換を適用するために、変換を初期化
        transform = self.color_jitter.get_params(self.color_jitter.brightness, self.color_jitter.contrast,
                                                 self.color_jitter.saturation, self.color_jitter.hue)

        # 各フレームに同じ変換を適用
        return torch.stack([transform(frame) for frame in video])

# シードの設定を行う関数
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def train(opt):

    # シードの設定
    seed_everything(opt.seed)

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # オプションで入力された値を取得
    csv_file = opt.data
    epochs = opt.epochs
    patience = opt.patience
    learning_rate = opt.lr
    batch = opt.batch

    # データの前処理
    transform = Compose([
        UniformTemporalSubsample(8),
        Lambda(lambda x: x / 255.0),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        Resize((256, 256)),
        VideoColorJitter(brightness=0.5)
    ])
    # データセットの作成
    video_dataset = VideoDataset(csv_file=csv_file, transform=transform)
    # データの分割
    train_size = int(0.7 * len(video_dataset))
    val_test_size = len(video_dataset) - train_size
    train_dataset, val_test_dataset = random_split(video_dataset, [train_size, val_test_size])
    test_size = int(0.33 * len(val_test_dataset))
    val_dataset, test_dataset = random_split(val_test_dataset, [test_size, len(val_test_dataset) - test_size])
    # データローダの作成
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # モデルの初期化
    model = CustomSlowFast(num_classes=1)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    calculate_model_size(model)

    # オプティマイザと損失関数
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion = BCEWithLogitsLoss()

    # 早期終了のための変数
    val_loss_min = None
    val_loss_min_epoch = 0

    # 結果を記録するための配列
    train_losses = []
    val_losses = []
    val_accuracy = []

    # 訓練ループ
    for epoch in range(epochs):
        # 訓練モード
        model.train()
        # エポックの損失をリセット
        train_loss = 0.0
        val_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配をゼロに
            optimizer.zero_grad()

            # 順伝播と逆伝播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 損失の取得
            train_loss += loss.item()

        # エポックごとの損失の計算
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # バリデーションモード
        model.eval()

        # accuracyの計算のための変数の初期化
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # モデルへの入力と出力の取得
                outputs = model(inputs)
                # 予測値の計算
                probs = torch.sigmoid(outputs)
                predicted = probs >= 0.5
                # accuracyの計算
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # 損失の計算
                val_loss += criterion(outputs, labels).item()
                
        # 損失関数の計算
        accuracy = 100 * correct / total
        # バリデーションロスの計算
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy.append(accuracy)

        print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

        # 早期終了の判定
        if val_loss_min is None or val_loss < val_loss_min:
            model_save_name = f'./latestresult/lr{learning_rate}_ep{epochs}_pa{patience}.pt'
            torch.save(model.state_dict(), model_save_name)
            val_loss_min = val_loss
            val_loss_min_epoch = epoch
        elif (epoch - val_loss_min_epoch) >= patience:
            # 早期終了
            print('Early stopping due to validation loss not improving for {} epochs'.format(patience))
            break

    # テストモード
    test_loss = []
    test_correct = 0
    test_total = 0  

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predicted = probs >= 0.5
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            cross_loss += criterion(outputs, labels).item()
            test_loss.append(cross_loss)

    # 損失関数の計算
    mean_test_loss = sum(test_loss) / len(test_loss)
    test_accuracy = 100 * test_correct / test_total
    print(f'Test MSE: {mean_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # グラフの描画
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

    return train_loss, val_loss_min

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print('start time:',start_time)
    sys.stdout.flush()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/labels.csv', help='path to data file')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr_search', type=str, default='false', help='whether to perform learning rate search')
    opt = parser.parse_args()

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

    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print('-----completing training-----')
    sys.stdout.flush()
    print('end time:',end_time)
    sys.stdout.flush()
    print('Execution time: ', execution_time)
    sys.stdout.flush()