import os
import random
import time

from setting import *
from captcha.image import ImageCaptcha
from PIL import Image

def generate_data(mode, count):
    """
    生成随机的图形验证码数据，注意重新运行前需要删掉旧数据集的目录
    :param mode: 模式，可选值：train/eval/predict/test
    :param count: 数量
    :return:
    """
    # 根据模式选择路径
    if mode == 'train':
        path = TRAIN_DATASET_PATH
    elif mode == 'eval':
        path = EVAL_DATASET_PATH
    elif mode == 'predict':
        path = PREDICT_DATASET_PATH
    elif mode == 'test':
        path = TEST_DATASET_PATH
    else:
        raise Exception('模式选择错误（train/eval/predict/test）')
    #
    os.makedirs(path, exist_ok=True)
    for i in range(count):
        # 随机生成图形验证码的文本
        captcha_text = ''
        for j in range(MAX_CAPTCHA):
            c = random.choice(ALL_CHAR_SET)
            captcha_text += c
        # print(f"本次产生的文本为：{captcha_text}")
        # 生成图形验证码
        image = ImageCaptcha(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        captcha_image = Image.open(image.generate(captcha_text))
        # 生成文件名并保存，注意相同文本的文件名可根据纳秒级时间戳进行区分
        timestamp = time.time_ns()
        image_name = f"{captcha_text}_{timestamp}.png"
        captcha_image.save(os.path.join(path, image_name))
        # print(f"图形验证码已生成：{image_name}")
        # 加一个打印进度
        if i % 1000 == 0:
            print(f"已生成 {i} 张图形验证码")
    print(f"{mode} 图形验证码，共 {count} 张已生成！")

generate_data('train', 100000)
