import torch
from model import CaptchaCNN
from dataset import get_predict_data_loader
from encoding import decode_predict, decode

def main():
    # 第1步：加载模型
    model = CaptchaCNN()
    model.load_state_dict(torch.load('model/best_model.pkl'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'正在使用设备：{device}')
    model = model.to(device)
    model.eval()

    # 第2步：准备测试集数据
    loader = get_predict_data_loader(batch_size=1)
    correct = 0
    total = 0

    # 第3步：遍历测试集，逐张预测并比较
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            # 取文件名
            file_path = loader.dataset.image_paths[i]
            file_name = file_path.split('\\')[-1]

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(0)
            # print(outputs)
            # print(labels)
            predict_text = decode_predict(outputs)
            true_text = decode(labels.squeeze(0))

            if predict_text == true_text:
                correct += 1
            total += 1

            print(f'{file_name} → 预测结果: {predict_text}')

    # 第4步：输出最终准确率
    print(f'测试完成！总准确率: {correct / total:.4f} ({correct}/{total})')

if __name__ == '__main__':
    main()
