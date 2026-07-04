import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Grayscale, ToTensor

from encoding import encode
from setting import TRAIN_DATASET_PATH, PREDICT_DATASET_PATH, EVAL_DATASET_PATH

class CaptchaDataset(Dataset):
    """数据集加载器"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # os.path.sep 表示系统路径分隔符
        file_name = os.path.basename(image_path)
        label_txt = file_name.split('_')[0]

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        label = encode(label_txt)
        return image, label

transform = transforms.Compose([
    # 转灰度，输出 1 通道
    Grayscale(num_output_channels=1),
    # 转为张量，形状 (C, H, W)，像素值 [0,1]
    ToTensor(),
])

def get_train_data_loader(batch_size=64):
    dataset = CaptchaDataset(TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_eval_data_loader(batch_size=1):
    dataset = CaptchaDataset(EVAL_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_predict_data_loader(batch_size=1):
    dataset = CaptchaDataset(PREDICT_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
