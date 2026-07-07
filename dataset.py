import os.path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Grayscale, ToTensor
from torch.utils.data import Dataset, DataLoader
from encoding import encode
from setting import *


class CaptchaDataset(Dataset):
    """
    验证码数据集

    PyTorch 的数据加载分两层：Dataset 负责"怎么读一张图"，DataLoader 负责"怎么打包成 batch、怎么打乱、怎么多线程读"
    """
    def __init__(self, data_dir, transform=None):
        # data_dir  : 数据集目录路径，如 'dataset/train'
        # transform : 图片预处理流水线（Grayscale → ToTensor）
        # 这里不读图片内容，只是记录路径。真正的读取发生在 __getitem__ 被调用时
        # 这样做的好处：10 万张图片不会一次性全部加载到内存，而是用到哪张读哪张
        self.data_dir = data_dir
        self.transform = transform
        # os.listdir(data_dir) 列出文件夹下所有文件名
        # os.path.join 拼接出完整路径
        # ['D:/.../dataset/train/BK7H_1612.png', 'D:/.../dataset/train/A0Z9_1613.png', ...]
        self.image_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]

    def __len__(self):
        # 返回数据集总样本数
        # DataLoader 用这个值来规划：一共几个 batch、一个 epoch 该跑多少步、shuffle 时索引范围是多少
        # 例如 10 万张图，batch_size=100 → DataLoader 知道每个 epoch 有 1000 个 batch
        return len(self.image_paths)

    def __getitem__(self, index):
        # 第1步：拿到图片的完整路径
        image_path = self.image_paths[index]
        # 第2步：从文件名解析标签，os.path.basename("D:/.../BK7H_1612080124.png") → "BK7H_1612080124.png"
        file_name = os.path.basename(image_path)
        # 切分结果: ["BK7H", "1612080124.png"]，取 [0] → "BK7H"
        label_txt = file_name.split('_')[0]
        # 第3步：打开图片（PIL Image），这张图片目前还是彩色的（RGB，3通道）
        image = Image.open(image_path)
        # 第4步：预处理（transform）
        if self.transform:
            image = self.transform(image)
        # 第5步：标签编码，encode("BK7H") → [0,0,...,1,...,0]  144 维 One-Hot 向量
        label = encode(label_txt)
        # 第6步：返回 (图片, 标签) 元组
        return image, label

# transform 是一条流水线，定义在文件末尾的全局变量里：
# Grayscale(num_output_channels=1)  → RGB转灰度（3通道→1通道）
# ToTensor()                        → PIL Image转Tensor + 像素值缩放到[0,1]
transform = transforms.Compose([
    Grayscale(1),
    ToTensor(),
])


def get_train_data_loader(batch_size=100):
    dataset = CaptchaDataset(TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_eval_data_loader(batch_size=1):
    dataset = CaptchaDataset(EVAL_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_predict_data_loader(batch_size=1):
    dataset = CaptchaDataset(PREDICT_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
