# create a model that predicts the disparity map
import torch
import torch.nn as nn
import torch.nn.functional as F

class StereoCNN(nn.Module):
    def __init__(self):
        super(StereoCNN, self).__init__()
        # Assuming input images are concatenated along the channel dimension
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)  # 6 channels input (2 images * 3 channels each)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)  # Output a single channel disparity map

    def forward(self, x):
        # x should be of shape [batch_size, 6, 400, 879] where 6 = 3 channels per stereo image
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.conv4(x)  # No activation, directly output the disparity values
        return x