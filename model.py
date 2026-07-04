import torch.nn as nn
from setting import *


class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        # 第一层卷积神经网络，共三层，每层都是 卷积->批标准化->随机失活->激活函数->池化 的流程
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 第二层卷积神经网络
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 第三层卷积神经网络
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 两个全连接层，将卷积层输出平铺后连接到全连接层
        self.fc = nn.Sequential(
            # 输入特征为图像缩小8倍后的长宽乘以64，输出特征为1024
            nn.Linear((IMAGE_WIDTH // 8) * (IMAGE_HEIGHT // 8) * 64, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, MAX_CAPTCHA * ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 将输出平铺为一维向量
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
