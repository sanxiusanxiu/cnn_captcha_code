import torch
from model import CaptchaCNN
from dataset import get_eval_data_loader
from encoding import decode_predict, decode

def main():
    # 第1步：加载训练好的模型
    model = CaptchaCNN()
    # load_state_dict：只加载参数，不加载模型结构（结构由 CaptchaCNN() 定义）
    # 这样做的好处：模型结构改了还能用旧参数（部分加载），灵活。坏处：必须保证保存时的结构和现在的完全一致，否则报错
    model.load_state_dict(torch.load('model/model.pkl'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'正在使用设备：{device}')

    # model.eval() 会改变 Dropout 和 BatchNorm 的行为：
    # Dropout：停止随机丢弃神经元 → 所有神经元都参与计算 → 输出稳定可复现
    # BatchNorm：不用当前 batch 的统计量，改用训练时积攒的全局均值/方差
    model = model.to(device)
    model.eval()

    # 第2步：准备验证集数据
    loader = get_eval_data_loader(batch_size=1)
    # 4 位全对的样本数
    correct = 0
    # 总样本数
    total = 0

    # 第3步：遍历验证集，逐张预测并比较
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            # images 形状: (batch=1, 1, 60, 160)
            # outputs 形状: (batch=1, 144)
            outputs = model(images)
            # squeeze(0)：去掉 batch 维度，(1, 144) → (144,)，因为 decode_predict() 期望输入是一维的 (144,) 向量
            outputs = outputs.squeeze(0)
            # 解码：144 维向量 → 4 位字符串
            predict_text = decode_predict(outputs)
            true_text = decode(labels.squeeze(0))

            if predict_text == true_text:
                correct += 1
            total += 1

            if total % 200 == 0:
                print(f'已评估 {total} 张，当前准确率: {correct/total:.4f}')

    # 第4步：输出最终准确率
    print(f'评估完成！总准确率: {correct/total:.4f} ({correct}/{total})')

if __name__ == '__main__':
    main()
