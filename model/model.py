import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet18
from torchsummary import summary
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1,1)):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # If the input and output channels differ, we need to use a 1x1 convolution to downsample the input
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If downsampling is necessary, apply the downsampling module
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out



class ResSigRegression(BaseModel):
    def __init__(self, out_pts, unit_size=50,last_activate='sigmoid'):
        super().__init__()

        self.last_activate = last_activate

            #self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding='same')

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ResidualBlock(64 ,64 ,stride=(1,1))
        self.layer2 = ResidualBlock(64, 64)
        self.layer3 = ResidualBlock(64, 128, stride=(2,2))
        self.layer4 = ResidualBlock(128, 128)
        self.layer5 = ResidualBlock(128, 256, stride=(2,2))
        self.layer6 = ResidualBlock(256, 256)

        self.avgpool = nn.AvgPool2d((3, 3))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.Linear(256, out_pts),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avgpool(x)
        x = self.fc(x)

        if self.last_activate == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.last_activate == 'regression':
            pass
        elif self.last_activate == 'tanh':
            x = torch.tanh(x)
        return x

class ResSmallRegression(BaseModel):
    def __init__(self, out_pts, unit_size=50,last_activate='sigmoid'):
        super().__init__()

        self.last_activate = last_activate

            #self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding='same')

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ResidualBlock(64 ,64 ,stride=(1,1))
        self.layer2 = ResidualBlock(64, 64)
        self.layer3 = ResidualBlock(64, 128, stride=(2,2))
        self.layer4 = ResidualBlock(128, 128)
        self.layer5 = ResidualBlock(128, 256, stride=(2,2))
        self.layer6 = ResidualBlock(256, 256)

        self.avgpool = nn.AvgPool2d((3, 3))
        self.fc = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(1024, 256),
            nn.Linear(1024, out_pts),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avgpool(x)
        x = self.fc(x)

        if self.last_activate == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.last_activate == 'regression':
            pass
        elif self.last_activate == 'tanh':
            x = torch.tanh(x)
        return x


#r=ResSigRegression(20,50,'sigmoid')
#summary(r, (1,50,50), batch_size=-1, device='cpu')