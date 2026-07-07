import os
import random
import sys
import time
from captcha.image import ImageCaptcha
from setting import ALL_CHAR_SET, MAX_CAPTCHA, TRAIN_DATASET_PATH, EVAL_DATASET_PATH
from PIL import Image

def generate_captcha_text():
    """生成随机验证码的文本"""
    captcha_text = []
    for i in range(MAX_CAPTCHA):
        c = random.choice(ALL_CHAR_SET)
        captcha_text.append(c)
    # 返回挑选的（随机）字符
    return ''.join(captcha_text)

def generate_captcha_text_and_image():
    """生成验证码图片，并返回文本和图像"""
    image = ImageCaptcha(width=160, height=60)
    captcha_text = generate_captcha_text()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image

def main(mode, count):
    """
    生成验证码图片
    mode: 'train' 或 'eval'
    count: 生成图片数量
    """
    if mode == 'train':
        save_path = TRAIN_DATASET_PATH
    elif mode == 'eval':
        save_path = EVAL_DATASET_PATH
    else:
        print("正确用法: python generate.py train 100000  或  python generate.py eval 30000")
        return

    os.makedirs(save_path, exist_ok=True)

    for i in range(count):
        text, image = generate_captcha_text_and_image()
        # 文件名 = 标签_时间戳.png
        timestamp = str(int(time.time() * 1000))
        filename = f'{text}_{timestamp}.png'
        image.save(os.path.join(save_path, filename))

        if (i + 1) % 1000 == 0:
            print(f'[{mode}] 已生成 {i+1}/{count} 张...')

    print(f'完成！共生成 {count} 张验证码到 {save_path}')

def generate_captcha_temp():
    """临时生成小批量的图形验证码"""
    temp_path = 'dataset/temp/'
    os.makedirs(temp_path, exist_ok=True)
    for i in range(300):
        now = str(int(time.time()))
        text, image = generate_captcha_text_and_image()
        filename = text + '_' + now + '.png'
        image.save(os.path.join(temp_path, filename))
        print('正在保存第 %d 张图片：%s' % (i + 1, filename))

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    # main('train', 100000)
    # main('eval', 30000)
    generate_captcha_temp()

