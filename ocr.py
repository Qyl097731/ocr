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

import Config as config
import tensorflow as tf
from keras import Input
from keras import layers
from scipy import ndimage
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from tensorflow.python.keras import models

import file_utils

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
    # 构建卷积神经网络
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad2D')(image_input)

    # 构建2D卷积层，来提取图像。
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
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
    return x


# 定义ResNet
def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def set_gpu_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)


def main(args):
    set_gpu_growth()
    # 载入标注文件
    annotation_files = file_utils.get_sub_files(config.TRAIN_LABEL_PATH)
    image_annotations = [reader.load_annotation(file, config.TRAIN_IMAGE_PATH) for file in annotation_files]
    # 删除错误的图像和标注路径
    image_annotations = [ann for ann in image_annotations if os.path.exists(ann['image_path'])]
    # 加载预训练模型，训练设置参数在 config.py 文件中
    m = models.ctpn_net(config, 'train')
    models.compile(m, config, loss_names=['ctpn_regress_loss', 'ctpn_class_loss',
                                          'side_regress_loss'])
    output = models.get_layer(m, 'ctpn_target').output
    models.add_metrics(m, ['gt_num', 'pos_num', 'neg_num', 'gt_min_iou', 'gt_avg_iou'],
                       output[-5:])
    # 训练时设置了 epochs 则采用，否则采用默认 epochs
    if args.init_epochs > 0:
        m.load_weights(args.weight_path, by_name=True)
    else:
        m.load_weights(config.PRE_TRAINED_WEIGHT, by_name=True)
    m.summary()
    # 设置生成器属性
    gen = generator(image_annotations[:-100], config.IMAGES_PER_GPU, config.IMAGE_SHAPE,
                    config.ANCHORS_WIDTH, config.MAX_GT_INSTANCES, horizontal_flip=False, random_crop=False)
    val_gen = generator(image_annotations[-100:], config.IMAGES_PER_GPU, config.IMAGE_SHAPE,
                        config.ANCHORS_WIDTH, config.MAX_GT_INSTANCES)
    # 开始训练
    m.fit_generator(gen,
                    steps_per_epoch=len(image_annotations) // config.IMAGES_PER_GPU * 2,
                    epochs=args.epochs,
                    initial_epoch=args.init_epochs,
                    validation_data=val_gen,
                    validation_steps=100 // config.IMAGES_PER_GPU,
                    verbose=True,
                    callbacks=get_call_back(),
                    workers=2,
                    use_multiprocessing=True)
    # 模型评估
    score = m.evaluate(gen, verbose=True)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


# 保存模

# def find_text_area():
#     # 覆盖默认参数
#     config.USE_SIDE_REFINE = bool(args.use_side_refine)
#     if args.weight_path is not None:
#         config.WEIGHT_PATH = args.weight_path
#         config.IMAGES_PER_GPU = args.images_per_gpu
#         config.IMAGE_SHAPE = args.image_shape
#         # 根据路径加载图片
#         image, image_meta, _, _ = image_utils.load_image_gt(np.random.randint(10),
#                                                             args.image_path,
#                                                             config.IMAGE_SHAPE[0],
#                                                             None)
#         # 加载模型
#         m = models.ctpn_net(config, "test")
#         m.load_weights(config.WEIGHT_PATH, by_name=True)
#         # 输出模型信息
#         m.summary()
#         # 模型预测
#         text_boxes, text_scores, _ = m.predict([np.array([image]),
#                                                 np.array([image_meta])])
#         text_boxes = np_utils.remove_pad(text_boxes[0])
#         text_scores = np_utils.remove_pad(text_scores[0])[:, 0]
#         # 文本行检测器
#         image_meta = image_utils.parse_image_meta(image_meta)
#         detector = TextDetector(config)
#         text_lines = detector.detect(text_boxes,
#                                      text_scores,
#                                      config.IMAGE_SHAPE,
#                                      image_meta['window'])
#         # 保存带检测框的图像
#         boxes_num = 30
#         fig = plt.figure(figsize=(16, 16))
#         ax = fig.add_subplot(1, 1, 1)
#         visualize.display_polygons(image, text_lines[:boxes_num, :8],
#                                    text_lines[:boxes_num, 8],
#                                    ax=ax)
#         image_name = os.path.basename(args.image_path)
#         fig.savefig('{}.{}.jpg'.format(os.path.splitext(image_name)[0],
#                                        int(config.USE_SIDE_REFINE)))


if __name__ == '__main__':
    image = img_initialize(imagePath)
    image_input = Input(shape=(224, 244, 1))
    image_input = resnet50(image_input=image_input)
