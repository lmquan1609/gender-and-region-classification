import torch
import torch.nn as nn
import torch.nn.functional as F



def num_flat_features(x):
    # (32L, 256L, 4L, 5L), 32 is batch_size
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class WaveMsNet(nn.Module):
    def __init__(self):
        super(WaveMsNet, self).__init__()

        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(32)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.bn1_3 = nn.BatchNorm1d(32)

        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(32)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.bn2_3 = nn.BatchNorm1d(32)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 8), stride=(3, 8))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 6)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchsize, 1, 3*16000)

        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x1 = self.pool2_1(x1)
        x2 = self.pool2_2(x2)
        x3 = self.pool2_3(x3) # (batchsize, 32, 320)

        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1) # (batchsize, 1, 32, 320)

        h = torch.cat((x1, x2, x3), dim=2) # (batchsize, 1, 96, 320)

        h = self.relu(self.bn3(self.conv3(h))) # (batchsize, 64, 96, 320)
        h = self.pool3(h) # (batchsize, 64, 32, 40)

        h = self.relu(self.bn4(self.conv4(h))) # (batchsize, 128, 32, 40)
        h = self.pool4(h) # (batchsize, 128, 16, 20)

        h = self.relu(self.bn5(self.conv5(h))) # (batchsize, 256, 16, 20)
        h = self.pool5(h) # (batchsize, 256, 8, 10)

        h = self.relu(self.bn6(self.conv6(h))) # (batchsize, 256, 8, 10)
        h = self.pool6(h) # (batchszie, 256, 4, 5)

        h = h.view(-1, num_flat_features(h)) # (batchsize, 5120)
        h = F.relu(self.fc1(h)) # (batchsize, 4096)
        h = self.dropout(h)
        h = self.fc2(h)

        return h