import torch
from PIL import Image
from torchvision.transforms import Grayscale, ToTensor, Compose, Resize
from model import CaptchaCNN
from encoding import decode_predict

def predict(path='captcha_picture.png'):
    # 第1步：加载模型
    model = CaptchaCNN()
    model.load_state_dict(torch.load('../model/best_model.pkl', map_location=torch.device('cpu')))
    model.eval()

    # 第2步：定义图片预处理流水线
    transform = Compose([Resize((60, 160)), Grayscale(1), ToTensor()])

    # 第3步：遍历图片，逐张识别
    # for file_name in os.listdir(PREDICT_DATASET_PATH):
    #     img = Image.open(os.path.join(PREDICT_DATASET_PATH, file_name))
    img = Image.open(path)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor).squeeze(0)
        pred_text = decode_predict(outputs)

    print(f'预测结果: {pred_text}')
    return pred_text
