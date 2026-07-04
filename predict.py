import torch
from PIL import Image
from torchvision.transforms import Grayscale, ToTensor, Compose
from model import CaptchaCNN
from encoding import decode_predict
from setting import PREDICT_DATASET_PATH
import os

def main():
    model = CaptchaCNN()
    model.load_state_dict(torch.load('model/best_model.pkl'))
    model.eval()

    transform = Compose([Grayscale(), ToTensor()])

    for filename in os.listdir(PREDICT_DATASET_PATH):
        img = Image.open(os.path.join(PREDICT_DATASET_PATH, filename))
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor).squeeze(0)
            pred_text = decode_predict(outputs)

        print(f'{filename} → 预测结果: {pred_text}')

if __name__ == '__main__':
    main()
