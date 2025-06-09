import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ViolenceDetector(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, num_classes=2, dropout_p=0.5):
        super(ViolenceDetector, self).__init__()
        self.FEATURE_SIZE = 512

        # CNN: Pretrained ResNet18 without final FC layer
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # LSTM
        self.lstm = nn.LSTM(self.FEATURE_SIZE, hidden_size, num_layers, batch_first=True)

        # Attention: learnable query vector
        self.attn_query = nn.Parameter(torch.empty(hidden_size, 1))
        nn.init.xavier_normal_(self.attn_query)

        # During training, 30% of the values will be randomly set to 0.
        self.dropout = nn.Dropout(p=dropout_p)

        # Fully connected classifier
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: (batch, time_steps, C, H, W)
        b, t, c, h, w = x.size()

        # CNN feature extraction for each frame
        cnn_features = []
        for i in range(t):
            f = self.cnn(x[:, i])  # (b, 512, 1, 1)
            f = f.view(b, 512)  # (b, 512)
            cnn_features.append(f)

        features = torch.stack(cnn_features, dim=1)  # (b, t, 512)

        # LSTM
        lstm_out, _ = self.lstm(features)  # (b, t, hidden_size)

        # Normalize LSTM output to help softmax gradients
        lstm_out = torch.nn.functional.normalize(lstm_out, p=2, dim=2)

        # Attention weights
        query = self.attn_query.unsqueeze(0).expand(b, -1, -1)  # (b, hidden_size, 1)
        attn_weights = torch.bmm(lstm_out, query).softmax(dim=1)  # (b, t, 1)

        # Apply attention weights to LSTM outputs
        attn_applied = torch.sum(lstm_out * attn_weights, dim=1)  # (b, hidden_size)

        dropout_applied = self.dropout(attn_applied)

        # Classification
        out = self.fc(dropout_applied)  # (b, num_classes)
        return out
