from torch import nn
from setting import *


class CaptchaCNN(nn.Module):
    """
    验证码识别模型

    整体结构：3 层卷积（提取图像特征） + 2 层全连接（分类输出）
    输入：1 * 160 * 60 的灰度验证码图片
    输出：144 维向量（4 个字符位置 * 每个位置 36 种字符的概率）
    """
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        # 第一层卷积，1×160×60 → 32×80×30
        # 第1层：「看线条」，识别边、角、端点
        self.conv1 = nn.Sequential(
            # Conv2d：卷积层，用 3×3 的卷积核在图片上滑动提取特征，输入 1 * 160 * 60，输出 32 * 160 * 60
            # in_channels=1   → 输入是灰度图，只有 1 个通道
            # out_channels=32 → 输出 32 个特征图（即用 32 个不同的卷积核各扫一遍）
            # kernel_size=3   → 每个卷积核大小 3×3
            # padding=1       → 边缘补一圈 0，保证卷积后尺寸不变（仍是 60×160）
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # BatchNorm2d：批量归一化，把每个通道的数据拉回均值 0、方差 1 附近
            # 作用：让训练更稳定、收敛更快，允许用更大的学习率
            nn.BatchNorm2d(32),
            # Dropout：训练时随机丢弃 50% 的神经元（置为 0）
            # 作用：防止模型过度依赖某些特定神经元，减少过拟合，每次训练都使用不同的"残缺网络"，强迫模型学到更鲁棒的特征
            nn.Dropout(0.5),
            # ReLU：激活函数，把所有负数变成 0，正数保持不变
            # 作用：引入非线性。没有激活函数的话，无论叠多少层卷积都等价于一层线性变换
            nn.ReLU(),
            # MaxPool2d：最大池化，2×2 窗口取最大值，尺寸减半，160×60 → 80×30
            # 作用：压缩特征图尺寸，减少计算量；保留最显著的特征，丢弃冗余信息
            nn.MaxPool2d(2),
        )
        # 第二层卷积，32×80×30 → 64×40×15
        # 第2层：「看笔画」，把边角拼成弧线、折线
        self.conv2 = nn.Sequential(
            # 通道数从 32 增加到 64，提取更丰富、更高级的特征（第一层找边缘/角点，第二层找笔画/弧线）
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 第三层卷积，64×40×15 → 64×20×7
        # 第3层：「看字形」，把笔画拼成字符部件
        self.conv3 = nn.Sequential(
            # 通道数保持 64 不变，但网络更深了（第三层找字符的整体形状/结构）
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            # 注意：15/2=7.5，PyTorch 的 MaxPool 默认向下取整，所以变成 7
            nn.MaxPool2d(2),
        )
        # 全连接：「做判断」，综合所有部件信号，判断每个位置是什么字
        self.fc = nn.Sequential(
            # 第一层全连接：8960 → 1024
            # 输入特征数 = 64(通道) × 7(高) × 20(宽) = 8960
            # IMAGE_WIDTH // 8 = 160 // 8 = 20
            # IMAGE_HEIGHT // 8 = 60 // 8 = 7（整除，和上面 MaxPool 的结果一致）
            # out_features=1024：压缩到 1024 维，减少参数量，降低过拟合风险
            nn.Linear(in_features=(IMAGE_WIDTH // 8) * (IMAGE_HEIGHT // 8) * 64, out_features=1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            # 第二层全连接（输出层）：1024 → 144
            # MAX_CAPTCHA * ALL_CHAR_SET_LEN = 4 × 36 = 144
            # 输出 144 个数字，对应 4 个字符位置各自的 36 种字符概率
            # 例如前 36 个数字是第 1 个字符的概率分布，接下来 36 个是第 2 个，依此类推
            nn.Linear(in_features=1024, out_features=MAX_CAPTCHA * ALL_CHAR_LIST_LEN),
        )

    def forward(self, x):
        # 前向传播：定义数据从输入到输出的流动路径，x 的形状：(batch_size, 1, 160, 60) → 一批灰度验证码图片
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 把三维特征图"拍平"成一维向量，才能喂给全连接层
        # x.size(0) 是 batch_size，-1 表示自动计算剩余维度，使得 x 从 (batch, 64, 7, 20) → (batch, 8960)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
