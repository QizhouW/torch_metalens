import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet18
from torchsummary import summary
import torch


class IdentityBlock(nn.Module):
    def __init__(self, input_channels, kernel_size, filters, stage, block):
        super(IdentityBlock, self).__init__()
        nb_filter1, nb_filter2 = filters

        self.conv_name_base = f"res{stage}{block}_branch"
        self.bn_name_base = f"bn{stage}{block}_branch"

        self.conv1 = nn.Conv2d(input_channels, nb_filter1, kernel_size, padding=(kernel_size // 2))
        self.bn1 = nn.BatchNorm2d(nb_filter1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nb_filter1, nb_filter2, kernel_size, padding=(kernel_size // 2))
        self.bn2 = nn.BatchNorm2d(nb_filter2)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x += shortcut
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_channels, kernel_size, filters, stage, block, strides=(2, 2)):
        super(ConvBlock, self).__init__()
        nb_filter1, nb_filter2 = filters

        self.conv_name_base = f"res{stage}{block}_branch"
        self.bn_name_base = f"bn{stage}{block}_branch"

        self.conv1 = nn.Conv2d(input_channels, nb_filter1, kernel_size, stride=strides, padding=(kernel_size // 2))
        self.bn1 = nn.BatchNorm2d(nb_filter1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nb_filter1, nb_filter2, kernel_size, padding=(kernel_size // 2))
        self.bn2 = nn.BatchNorm2d(nb_filter2)

        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channels, nb_filter2, kernel_size=(1, 1), stride=strides),
            nn.BatchNorm2d(nb_filter2)
        )

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x += shortcut
        x =self.relu(x)
        return x



class ResSigRegression(BaseModel):
    def __init__(self, input_shape, output_shape, last_activate='sigmoid'):
        super().__init__()

        self.last_activate = last_activate

        if input_shape== 100:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding='same')
        elif input_shape == 50:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding='same')

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ConvBlock(64, 3, [64, 64], stage=2, block='a', strides=(1, 1))
        self.layer2 = IdentityBlock(64, 3, [64, 64], stage=2, block='b')
        self.layer3 = ConvBlock(64, 3, [128, 128], stage=3, block='a')
        self.layer4 = IdentityBlock(128, 3, [128, 128], stage=3, block='b')
        self.layer5 = ConvBlock(128, 3, [256, 256], stage=4,  block='a')
        self.layer6 = IdentityBlock(256, 3, [256, 256], stage=4, block='b')

        self.avgpool = nn.AvgPool2d((3, 3))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, output_shape),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

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


#r=ResSigRegression((1,50,50),20,'relu')
#summary(r, (1,50,50), batch_size=-1, device='cpu')