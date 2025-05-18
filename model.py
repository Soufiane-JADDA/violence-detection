import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ViolenceDetector(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, num_classes=2):
        super(ViolenceDetector, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
        self.lstm = nn.LSTM(512, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: (batch, time_steps, C, H, W)
        b, t, c, h, w = x.size()
        cnn_features = []
        for i in range(t):
            f = self.cnn(x[:, i])  # shape: (b, 512, 1, 1)
            f = f.view(b, 512)     # Flatten
            cnn_features.append(f)
        features = torch.stack(cnn_features, dim=1)  # shape: (b, t, 512)
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])  # Take last output
        return out