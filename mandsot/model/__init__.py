import torch
import torch.nn as nn


class MandSOT(nn.Module):
    def __init__(self):
        super(MandSOT, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=80, out_channels=128, kernel_size=2, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.2)
        # self.embedding_layer = nn.Embedding(num_embeddings=30, embedding_dim=64)

        self.fc1 = nn.Linear(in_features=36096, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=1)

    def forward(self, mfcc, initial):
        x = self.pool1(torch.relu(self.conv1(mfcc)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))

        # x = torch.cat([x, x1, x2, x3], 2)
        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
