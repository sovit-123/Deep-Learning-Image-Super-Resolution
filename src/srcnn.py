import torch.nn as nn
import torch.nn.functional as F

class SRCNN1(nn.Module):
    def __init__(self):
        super(SRCNN1, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class SRCNN2(nn.Module):
    def __init__(self):
        super(SRCNN2, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 3, kernel_size=5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class SRCNN3(nn.Module):
    def __init__(self):
        super(SRCNN3, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv5 = nn.Conv2d(64, 3, kernel_size=5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        return x


class SRCNN4(nn.Module):
    def __init__(self):
        super(SRCNN4, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, padding=3, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=3, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 3, kernel_size=7, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x