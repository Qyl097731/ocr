# _*_ coding: utf-8 _*_
# @Time : 2023/5/20 21:22
# @Author : 邱依良
# @Version：V 0.1
# @File : crnn_keras.py
# @desc : 文字识别模块

import cv2
from PIL import Image
from crnn.utils import StrLabelConverter, ResizeNormalize
from crnn import keys
from config import ocrModelKeras
import numpy as np
from crnn.network_keras import keras_crnn as crnn


def crnn_source():
    alphabet = keys.alphabetChinese  # 中英文模型
    converter = StrLabelConverter(alphabet)
    model = crnn(32, 1, len(alphabet) + 1, 256, 1)
    model.load_weights(ocrModelKeras)
    return model, converter


# 加载模型
model, converter = crnn_source()


def crnn_ocr(image):
    """
    crnn模型，ocr识别
    image:PIL.Image.convert("L")
    """
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    transformer = ResizeNormalize((w, 32))
    image = transformer(image)
    image = image.astype(np.float32)
    image = np.array([[image]])
    preds = model.predict(image)
    preds = preds[0]
    preds = np.argmax(preds, axis=2).reshape((-1,))
    sim_pred = converter.decode(preds)
    return sim_pred


if __name__ == '__main__':
    img = cv2.imread('D:\python_demo\ocr\data\\crnn_test.png')
    partImg = Image.fromarray(img)
    text = crnn_ocr(partImg.convert('L'))
    print(text)
