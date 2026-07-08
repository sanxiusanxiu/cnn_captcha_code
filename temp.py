import torch

# 测试编码解码是否生效
def test_encoding():
    from encoding import encode, decode
    # 输出是一个 24*6 的矩阵，其中 12 对应着字符 'B'
    print(encode('BK7H'))
    print(decode(encode('BK7H')))


if __name__ == '__main__':
    # test_encoding()
    # !nvidia-smi
    # CUDA Version: 13.2
    # !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'正在使用设备：{device}')

