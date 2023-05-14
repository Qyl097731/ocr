# _*_ coding: utf-8 _*_
# @Time : 2023/5/14 14:15
# @Author : 邱依良
# @Version：V 0.1
# @File : ocr.py
# @desc : ocr 封装

import os
import cv2
import numpy as np
import math

from keras.utils import np_utils, image_utils
from scipy import ndimage
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.lite.tools import visualize
from tensorflow.python.keras import models
import matplotlib.pyplot as plt


imagePath = './data/form.png'


# 图像预处理
def img_initialize(image_path):
    # 霍夫变换 倾斜矫正
    def img_rotate():
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

        # 霍夫变换
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
        rotate_angle = 0
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 - 1000 * b)
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 + 1000 * b)
            y2 = int(y0 - 1000 * a)
            if x1 == x2 or y1 == y2:
                continue
            t = float(y2 - y1) / (x2 - x1)
            rotate_angle = math.degrees(math.atan(t))
            if rotate_angle > 45:
                rotate_angle = -90 + rotate_angle
            elif rotate_angle < -45:
                rotate_angle = 90 + rotate_angle
        return ndimage.rotate(binary_image, rotate_angle)

    image = cv2.imread(image_path)
    # 灰度处理
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    binary_image = cv2.adaptiveThreshold(~gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -5)
    # 倾斜矫正
    binary_image = img_rotate()

    cv2.imshow("图像初始化", binary_image)
    cv2.waitKey(0)
    return binary_image


# ResNet 网络骨架替换
def resnet50(image_input):
    bn_axis = 3
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_Pad2D')(image_input)
    # stage 1
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    no_train_model = keras.Model(inputs=image_input, outputs=x)
    for l in no_train_model.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = True
        else:
            l.trainable = False
    # stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    # 采用 conv4 特征图做输出，不需要 stage5
    # stage 5
    return


def find_text_area():
    # 覆盖默认参数
    config.USE_SIDE_REFINE = bool(args.use_side_refine)
    if args.weight_path is not None:
        config.WEIGHT_PATH = args.weight_path
        config.IMAGES_PER_GPU = args.images_per_gpu
        config.IMAGE_SHAPE = args.image_shape
        # 根据路径加载图片
        image, image_meta, _, _ = image_utils.load_image_gt(np.random.randint(10),
                                                            args.image_path,
                                                            config.IMAGE_SHAPE[0],
                                                            None)
        # 加载模型
        m = models.ctpn_net(config, "test")
        m.load_weights(config.WEIGHT_PATH, by_name=True)
        # 输出模型信息
        m.summary()
        # 模型预测
        text_boxes, text_scores, _ = m.predict([np.array([image]),
                                                np.array([image_meta])])
        text_boxes = np_utils.remove_pad(text_boxes[0])
        text_scores = np_utils.remove_pad(text_scores[0])[:, 0]
        # 文本行检测器
        image_meta = image_utils.parse_image_meta(image_meta)
        detector = TextDetector(config)
        text_lines = detector.detect(text_boxes,
                                     text_scores,
                                     config.IMAGE_SHAPE,
                                     image_meta['window'])
        # 保存带检测框的图像
        boxes_num = 30
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(1, 1, 1)
        visualize.display_polygons(image, text_lines[:boxes_num, :8],
                                   text_lines[:boxes_num, 8],
                                   ax=ax)
        image_name = os.path.basename(args.image_path)
        fig.savefig('{}.{}.jpg'.format(os.path.splitext(image_name)[0],
                                       int(config.USE_SIDE_REFINE)))


# 定义ResNet
def identity_block(inputs, filters):
    filters1, filters2, filters3 = filters

    x = layers.Conv2D(filters1,
                      (1, 1),
                      padding='valid')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2,
                      (3, 3),
                      padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3,
                      (1, 1),
                      padding='valid')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, inputs])
    x = layers.Activation('relu')(x)
    return x


def conv_block(inputs, filters, strides):
    filters1, filters2, filters3 = filters

    x = layers.Conv2D(filters1,
                      (1, 1),
                      strides=strides,
                      padding='valid')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2,
                      (3, 3),
                      padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3,
                      (1, 1),
                      padding='valid')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(filters3,
                             (1, 1),
                             strides=strides,
                             padding='valid')(inputs)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


img_initialize(imagePath)
