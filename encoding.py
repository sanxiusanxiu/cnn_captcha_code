import numpy as np
from setting import ALL_CHAR_SET, ALL_CHAR_SET_LEN, MAX_CAPTCHA

# 字符-索引 映射表
char_to_idx = {c: i for i, c in enumerate(ALL_CHAR_SET)}  # {'0':0, '1':1, ..., 'Z':35}
idx_to_char = {i: c for i, c in enumerate(ALL_CHAR_SET)}  # {0:'0', 1:'1', ..., 35:'Z'}

def encode(text):
    """把 'BK7H' 编码为 144 维 One-Hot 向量"""
    # 4 * 36 = 144 个 0
    vec = np.zeros(MAX_CAPTCHA * ALL_CHAR_SET_LEN)
    for i, char in enumerate(text):
        # 字符对应的位置
        idx = char_to_idx[char]
        # 注意第 i 个字符的位置范围: [i * 36, (i + 1) * 36)
        position = i * ALL_CHAR_SET_LEN + idx
        # 将字符对应的位置值置为 1
        vec[position] = 1
    return vec

def decode(vec):
    """把 144 维 One-Hot 向量解码回字符串"""
    chars = []
    for i in range(MAX_CAPTCHA):
        # 截取第 i 个字符对应的片段
        start = i * ALL_CHAR_SET_LEN
        end = (i + 1) * ALL_CHAR_SET_LEN
        segment = vec[start : end]
        # 找到值为 1 的位置（取最大值的索引）
        idx = np.argmax(segment)
        chars.append(idx_to_char[idx])
    return ''.join(chars)

"""
# decode_predict(outputs)：模型输出是连续概率值（0~1 的小数）
#   内部：每 36 个一组，找概率最大的那个 → 字符
#   例：[0.01, 0.02, ..., 0.92, ..., 0.005] → 索引 11 → 'B'
#
# decode(labels)：标签是 One-Hot 编码（一个 1，其余全是 0）
#   内部：每 36 个一组，找值为 1 的那个 → 字符
#   例：[0, 0, ..., 1, ..., 0] → 索引 11 → 'B'
"""

def decode_predict(output_vec):
    """从模型输出的 144 维概率向量解码为字符串（无需 One-Hot，直接取每段 argmax）"""
    chars = []
    # 概率向量
    for i in range(MAX_CAPTCHA):
        start = i * ALL_CHAR_SET_LEN
        end = (i + 1) * ALL_CHAR_SET_LEN
        segment = vec[start: end]
        idx = np.argmax(segment)
        chars.append(idx_to_char[idx])
    return ''.join(chars)
