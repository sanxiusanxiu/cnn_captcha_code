# 测试编码解码是否生效
def test_encoding():
    from encoding import encode, decode
    # 输出是一个 24*6 的矩阵，其中 12 对应着字符 'B'
    print(encode('BK7H'))
    print(decode(encode('BK7H')))


if __name__ == '__main__':
    test_encoding()

