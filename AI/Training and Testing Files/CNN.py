# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)

#         identity = self.downsample(identity)

#         out += identity
#         out = self.relu(out)

#         return out

# class EmotionCNN(nn.Module):
#     def __init__(self, num_classes=7):
#         super(EmotionCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.resblock1 = ResidualBlock(64, 64)
#         self.resblock2 = ResidualBlock(64, 128, stride=2)
#         self.resblock3 = ResidualBlock(128, 128)
#         self.resblock4 = ResidualBlock(128, 256, stride=2)
#         self.resblock5 = ResidualBlock(256, 256)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.resblock1(x)
#         x = self.resblock2(x)
#         x = self.resblock3(x)
#         x = self.resblock4(x)
#         x = self.resblock5(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

#----------------------------------#

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class EmotionCNN(nn.Module):
#     def __init__(self, num_classes=7):
#         super(EmotionCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)

#         # Reduced number of convolutional blocks from 8 to 5
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Reduced number of fully connected layers from 3 to 1
#         self.fc1 = nn.Linear(256, num_classes)

#         # Added dropout layer with a probability of 0.5 after the fully connected layer
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.maxpool1(x)

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.maxpool2(x)

#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu(x)
#         x = self.maxpool3(x)

#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.relu(x)
#         x = self.maxpool4(x)

#         x = x.view(x.size(0), -1)

#         x = self.fc1(x)
#         x = self.dropout(x)

#         return x


# ----------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Reduced number of convolutional blocks from 8 to 5
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Reduced number of fully connected layers from 3 to 1
        self.fc1 = nn.Linear(256 * 3 * 3, num_classes)  # Adjusted input size

        # Added dropout layer with a probability of 0.5 after the fully connected layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool4(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.dropout(x)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class DeeperEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(DeeperEmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjusted the input size of fc1 based on the printed size before the fully connected layer
        self.fc1 = nn.Linear(512 * 1 * 1, num_classes)

        # Added dropout layer with a probability of 0.5 after the fully connected layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool4(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.maxpool5(x)

        x = x.view(x.size(0), -1)

        # Print the size of x before the fully connected layer
        # print("Size before FC:", x.size())

        x = self.fc1(x)
        x = self.dropout(x)

        return x
