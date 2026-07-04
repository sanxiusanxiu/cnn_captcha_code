import numpy as np
from setting import ALL_CHAR_SET, ALL_CHAR_SET_LEN, MAX_CAPTCHA

# 字符-索引 映射表
char_to_idx = {c: i for i, c in enumerate(ALL_CHAR_SET)}  # {'0':0, '1':1, ..., 'Z':35}
idx_to_char = {i: c for i, c in enumerate(ALL_CHAR_SET)}  # {0:'0', 1:'1', ..., 35:'Z'}

def encode(text):
    """把 'BK7H' 编码为 144 维 One-Hot 向量"""
    # 144 个 0
    vec = np.zeros(MAX_CAPTCHA * ALL_CHAR_SET_LEN)
    for i, char in enumerate(text):
        # 字符对应的位置
        idx = char_to_idx[char]
        # 对应位置置 1
        vec[i * ALL_CHAR_SET_LEN + idx] = 1
    return vec

def decode(vec):
    """把 144 维 One-Hot 向量解码回字符串"""
    chars = []
    for i in range(MAX_CAPTCHA):
        segment = vec[i * ALL_CHAR_SET_LEN : (i + 1) * ALL_CHAR_SET_LEN]
        # 找到值为1的位置
        idx = np.argmax(segment)
        chars.append(idx_to_char[idx])
    return ''.join(chars)

# ⭐ 优化新增：从模型输出（概率值）直接解码，evaluate 和 predict 共享
def decode_predict(output_vec):
    """从模型输出的 144 维概率向量解码为字符串（无需 One-Hot，直接取每段 argmax）"""
    chars = []
    for i in range(MAX_CAPTCHA):
        segment = output_vec[i * ALL_CHAR_SET_LEN : (i + 1) * ALL_CHAR_SET_LEN]
        idx = np.argmax(segment)
        chars.append(idx_to_char[idx])
    return ''.join(chars)
