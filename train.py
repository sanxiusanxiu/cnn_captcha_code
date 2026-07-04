import torch
import torch.nn as nn
import numpy as np

import model
from model import CaptchaCNN
from dataset import get_train_data_loader, get_eval_data_loader
from encoding import decode_predict, decode
from setting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
num_epochs = 3

def evaluate_model():
    eval_loader = get_eval_data_loader(batch_size=1)
    model.eval().to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in eval_loader:
            # (batch, 4, 36)
            outputs = outputs.view(-1, MAX_CAPTCHA, ALL_CHAR_SET_LEN)
            # (batch, 4)
            outputs = outputs.argmax(dim=2)

            labels = labels.view(-1, MAX_CAPTCHA, ALL_CHAR_SET_LEN)
            labels = labels.argmax(dim=2)

            # 4个字符全部预测正确才算对（完全匹配）
            correct += (outputs == labels).all(dim=1).sum().item()
            total += images.size(0)

    # 恢复训练模式
    model.train().to(device)
    return correct / total


def train():
    train_loader = get_train_data_loader(batch_size=64)
    eval_loader = get_eval_data_loader(batch_size=1)

    model = CaptchaCNN().to(device)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train().to(device)
        total_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}] Step [{i}] Loss: {loss.item():.4f}')

        # 每个 epoch 结束后的评估
        accuracy = evaluate_model(model, eval_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] 平均Loss: {total_loss / len(train_loader):.4f} 验证准确率: {accuracy:.4f}')

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), './model/best_model.pkl')
            print(f'  → 新的最佳模型！准确率: {best_accuracy:.4f}')

    # 保存最终模型
    torch.save(model.state_dict(), './model/model.pkl')
    print(f'训练完成！最终模型已保存，其中最佳模型准确率: {best_accuracy:.4f}')

if __name__ == '__main__':
    train()
