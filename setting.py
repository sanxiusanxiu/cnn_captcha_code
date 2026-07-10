NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

ALL_CHAR_LIST = NUMBER + ALPHABET

ALL_CHAR_LIST_LEN = len(ALL_CHAR_LIST)
# 验证码固定4位
MAX_CAPTCHA = 4

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
# 数据集存放路径
TRAIN_DATASET_PATH = 'dataset/train'
EVAL_DATASET_PATH = 'dataset/eval'
PREDICT_DATASET_PATH = 'dataset/predict'
TEST_DATASET_PATH = 'dataset/test'
