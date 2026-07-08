import os
import torch
from torch import nn
from model import CaptchaCNN
from dataset import *

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'正在使用设备：{device}')

os.mkdir('model')

# 超参数
num_epochs = 30

def evaluate_model(model, eval_loader):
    model = model.to(device)
    model.eval()
    # 4 位全对的样本数
    correct = 0
    # 总样本数
    total = 0
    # torch.no_grad()：告诉 PyTorch "这段代码不需要算梯度"，可以省显存（不存中间结果）、省时间（跳过反向传播的准备）
    # 评估/推理时必须加，否则跑几万张图显存就炸了
    with torch.no_grad():
        for i, (images, labels) in enumerate(eval_loader):
            images, labels = images.to(device), labels.to(device)
            # 前向传播：图片 → 模型 → 144 维预测向量，因为 batch_size=1，所以 outputs 形状为 (batch_size, 144)
            outputs = model(images)

            # 解码预测结果
            # 把 144 维向量(batch=1, 144) reshape 成 (batch=1, 4个字符, 36种字符)
            outputs = outputs.view(-1, MAX_CAPTCHA, ALL_CHAR_SET_LEN)
            # 然后 dim=2 表示取维度 2，argmax(dim=2)：每个字符位置取 36 个概率里最大的那个
            # outputs 形状 (1, 4)：[[11, 20, 7, 17]] → 4 个字符在字符集里的索引 → 'B','K','7','H'
            outputs = outputs.argmax(dim=2)

            # 解码真实标签
            # labels 是 One-Hot 编码的 144 维向量
            # 同样 reshape + argmax 得到字符索引
            labels = labels.view(-1, MAX_CAPTCHA, ALL_CHAR_SET_LEN)
            labels = labels.argmax(dim=2)

            # 统计完全匹配的样本数
            # outputs == labels：逐位置比较，得到 True/False 矩阵
            # .all(dim=1)：只有 True/False 矩阵的 4 个位置全部为 True 才得出 True
            # .sum().item()：True 记为 1，Tensor 转数字
            correct = (outputs == labels).all(dim=1).sum().item()
            # labels 的形状是 (batch_size, 4)
            total += labels.size(0)

    model.train()
    return correct / total

def train():
    # 第1步：准备数据
    train_loader = get_train_data_loader(batch_size=100)
    eval_loader = get_eval_data_loader(batch_size=1)
    # 第2步：初始化模型
    model = CaptchaCNN().to(device)

    # 第3步：定义损失函数
    # MultiLabelSoftMarginLoss：多标签分类损失，内部原理（对 144 个输出位置逐个计算）：
    #    ① Sigmoid：把 [-∞, +∞] 映射到 [0, 1]
    #    ② BCELoss：比较预测值和真实标签（0 或 1）
    #    ③ 对 144 个位置求平均 → 最终的 loss 值
    #
    # 为什么不用 CrossEntropyLoss？因为 CrossEntropyLoss 假设所有类别互斥（比如一只动物不能同时是猫和狗）
    # 但验证码有 4 个独立位置，第1位是 'B' ≠ 第2位不能是 'K'，所以需要多标签分类：每个位置独立判断，互不干扰
    criterion = nn.MultiLabelSoftMarginLoss()

    # 第4步：选择优化器
    # Adam：自适应学习率优化器，相对于普通 SGD 多做两件事：
    #    ① Momentum（动量）：记住之前的更新方向，加速收敛
    #    ② Adaptive LR：每个参数有自己的学习率，用得少的参数用大步调
    # model.parameters()：告诉优化器"你要管的是模型里所有可训练参数"
    # lr=0.001：学习率常用默认值，对这个任务来说够用
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 第5步：记录历史最佳准确率
    best_accuracy = 0.0

    # 第6步：训练循环
    for epoch in range(num_epochs):
        model = model.to(device)
        model.train()
        # 当前轮次下所有 batch 的 loss
        total_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播（Forward），图片经过三层卷积 + 两层全连接，得到 144 维预测向量
            # images 形状: (batch_size, 1, 60, 160)；outputs 形状: (batch_size, 144)
            outputs = model(images)

            # 计算损失（Loss）
            # criterion(outputs, labels) 内部流程：对每个位置做 Sigmoid → 和 float 标签比较 → 用 BCELoss 算差距
            # 因为 MultiLabelSoftMarginLoss 内部用的是 BCELoss，BCELoss 需要浮点型标签，所以要 labels.float()
            loss = criterion(outputs, labels.float())

            # 反向传播（Backward）
            # 注意 zero_grad() 必须在 backward() 之前
            # PyTorch 默认累积梯度（累加而非覆盖），如果不先清零，新的梯度会叠在旧梯度上，参数更新幅度会越来越离谱
            optimizer.zero_grad()
            # 从 loss 出发，沿网络反向传播，自动算出每个参数的梯度。用到的数学原理就是链式法则（复合函数求导）
            loss.backward()

            # 参数更新（Step）
            # 优化器根据算好的梯度调整参数
            # 另外，Adam 在这一步还做了：用动量平滑梯度（减少震荡）；每个参数单独调整学习率
            optimizer.step()

            # 计算损失值
            total_loss += loss.item()
            if i % 100 == 0:
                print(f'轮次：[{epoch + 1}/{num_epochs}] Step[{i}] Loss: {loss.item() :.4f}')

        # 每个 epoch 结束后进行评估
        accuracy = evaluate_model(model, eval_loader)
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] 平均Loss: {avg_loss:.4f} 验证准确率: {accuracy:.4f}')

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), './model/best_model.pkl')
            print(f'  → 新的最佳模型！准确率: {best_accuracy:.4f}')

    # 保存最终模型（不一定是最佳的，但可能是泛化最好的一般模型）
    torch.save(model.state_dict(), './model/model.pkl')
    print(f'训练完成！最终模型已保存，最佳模型准确率: {best_accuracy:.4f}')

if __name__ == '__main__':
    train()
