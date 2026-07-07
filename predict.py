import torch
from PIL import Image
from torchvision.transforms import Grayscale, ToTensor, Compose
from model import CaptchaCNN
from encoding import decode_predict
from setting import PREDICT_DATASET_PATH
import os

def main():
    # 第1步：加载模型
    model = CaptchaCNN()
    model.load_state_dict(torch.load('model/best_model.pkl'))
    model.eval()

    # 第2步：定义图片预处理流水线
    transform = Compose([Grayscale(), ToTensor()])

    # 第3步：遍历预测集，逐张识别
    for file_name in os.listdir(PREDICT_DATASET_PATH):
        img = Image.open(os.path.join(PREDICT_DATASET_PATH, file_name))
        # 预处理 + 加 batch 维度
        # transform(img) 形状: (1, 60, 160)
        # unsqueeze(0)：在第 0 维加一个维度
        #   (1, 60, 160) → (1, 1, 60, 160)
        #                    ↑
        #                  batch 维度
        img_tensor = transform(img).unsqueeze(0)

        # 前向传播 + 解码
        with torch.no_grad():
            # model(img_tensor) 形状: (1, 144)
            # .squeeze(0) 去掉 batch 维度: (144,)
            # decode_predict：144 维概率向量 → 4 位字符串
            outputs = model(img_tensor).squeeze(0)
            pred_text = decode_predict(outputs)

        print(f'{file_name} → 预测结果: {pred_text}')

if __name__ == '__main__':
    main()
