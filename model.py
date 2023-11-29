import torch
import torch.nn as nn
import torch.hub

class CustomSlowFast(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomSlowFast, self).__init__()
        # PyTorch Hubから事前学習されたモデルをロード
        self.slowfast = torch.hub.load(
            'facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True
        )
        # 最終分類層の入力特徴量数を取得
        in_features = self.slowfast.head.projection.in_features
        # ２値分類用の線形層に置き換え
        self.slowfast.head.projection = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.slowfast(x)