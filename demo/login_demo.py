import time
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from predict_one import predict

# CNN训练的图形验证码模型 效果测试

def login():
    try:
        # https://captcha7.scrape.center/
        # https://captcha8.scrape.center/
        url = 'https://captcha8.scrape.center/'
        driver.get(url)
        time.sleep(3)
        # 找到输入框
        inputs = driver.find_elements(By.TAG_NAME, 'input')
        time.sleep(1)
        inputs[0].send_keys('admin')
        time.sleep(1)
        inputs[1].send_keys('admin')
        time.sleep(1)
        # 保存图形验证码
        canvas = driver.find_element(By.TAG_NAME, 'canvas')
        canvas_base64 = driver.execute_script(
            "return arguments[0].toDataURL('image/png').substring(21);",
            canvas
        )
        img_data = base64.b64decode(canvas_base64)
        with open('captcha_picture.png', 'wb') as f:
            f.write(img_data)
        time.sleep(1)
        # 使用模型进行预测
        predict_text = predict()
        # print(f'模型预测结果: {predict_text}')
        inputs[2].send_keys(predict_text)
        time.sleep(1)
        # 找到登录按钮
        button = driver.find_element(By.TAG_NAME, 'button')
        button.click()
        time.sleep(3)
    finally:
        driver.quit()

if __name__ == '__main__':
    # 设置 Chrome 选项
    chrome_driver_path = 'chromedriver.exe'
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    service = Service(executable_path=chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    #
    login()