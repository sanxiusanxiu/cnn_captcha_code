import numpy as np
from setting import *

# 创建 字符-索引 映射表
char_to_index = {}
for i, c in enumerate(ALL_CHAR_SET):
    char_to_index[c] = i
# {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}
# print(char_to_index)
# 创建 索引-字符 映射表
index_to_char = {}
for i, c in enumerate(ALL_CHAR_SET):
    index_to_char[i] = c
# {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}
# print(index_to_char)

def encode(text):
    """编码，如把 'BK7H' 编码为 4*36=144 维的 One-Hot 向量"""
    vector = np.zeros(MAX_CAPTCHA * ALL_CHAR_SET_LEN)
    for i, char in enumerate(text):
        # 根据字符得到索引
        index = char_to_index[char]
        # 注意第 i 个字符的位置范围是 [i * 36, (i + 1) * 36)
        position = i * ALL_CHAR_SET_LEN + index
        # 将字符对应的位置值置为 1
        vector[position] = 1
    return vector

def decode(vector):
    """
    解码，如把 144 维的 One-Hot 向量解码为字符串
    每 36 个一组，找值为 1 的那个 → 字符。例：[0, 0, ..., 1, ..., 0] → 索引 11 → 'B'
    """
    chars = []
    for i in range(MAX_CAPTCHA):
        # 截取第 i 个字符对应的片段
        start = i * ALL_CHAR_SET_LEN
        end = (i + 1) * ALL_CHAR_SET_LEN
        segment = vector[start: end]
        # 找到值为 1 的位置（在这个稀疏矩阵中最大值就是 1）
        index = np.argmax(segment)
        chars.append(index_to_char[index])
    return ''.join(chars)

def decode_predict(output_vector):
    """
    从模型输出的 144 维概率向量解码为字符串
    每 36 个一组，找概率最大的那个 → 字符。例：[0.01, 0.02, ..., 0.92, ..., 0.005] → 索引 11 → 'B'
    """
    chars = []
    # 概率向量
    for i in range(MAX_CAPTCHA):
        start = i * ALL_CHAR_SET_LEN
        end = (i + 1) * ALL_CHAR_SET_LEN
        segment = output_vector[start: end]
        index = np.argmax(segment)
        chars.append(index_to_char[index])
    return ''.join(chars)
