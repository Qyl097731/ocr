# _*_ coding: utf-8 _*_
# @Time : 2023/5/14 14:15
# @Author : 邱依良
# @Version：V 0.1
# @File : ocr.py
# @desc : ocr 封装

import os
from collections import generator

import cv2
import numpy as np
import math

import yaml
import argparse
import tensorflow as tf
from keras import Input
from keras import layers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy import ndimage
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, LSTM, concatenate, \
    TimeDistributed
from keras.layers import Activation, BatchNormalization, Flatten
from tensorflow.python.keras import models

import file_utils

imagePath = './data/form.png'


def dict2namespace(config):
    """将字典转换为命名空间对象"""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def load_annotation(annotation_file, image_dir):
    """ 从标注文件加载图像与标注框信息 """
    with open(annotation_file, 'r') as f:
        content = f.readlines()
    # 跳过前两行注释
    content = content[2:]
    annotation = {'image_path': '', 'boxes': []}
    for line in content:
        line_parts = line.split(',')
        if len(line_parts) == 1:  # 存储 image_path
            annotation['image_path'] = os.path.join(image_dir, line.strip())
        else:  # 存储标注框信息
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, line_parts[:8])
            annotation['boxes'].append([x1, y1, x2, y2, x3, y3, x4, y4])
    return annotation


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


# 第一步 ResNet 网络骨架替换 得到的 conv5_x 特征作为特征图像（feature map），输出大小为 14×14×1024；
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


# 第二步 将一个 3x3 大小的卷积核在 x 上进行滑动，得到一个 14x14x1024 大小的输出。
def extract_feature_vector(features):
    # 定义滑动窗口的参数
    kernel_size = (3, 3)

    # 定义滑动窗口的卷积层
    conv_layer = Conv2D(filters=1024,
                        kernel_size=kernel_size,
                        padding='same',
                        activation='relu')

    # 在特征图像上进行滑动窗口操作
    feature_vector = conv_layer(features)

    return feature_vector


# 第三步，双向LSTM
def feed_to_rnn(features):
    # 定义一个正向 LSTM 层，包含 128 个隐层，并返回完整的输出序列
    lstm_fw = LSTM(units=128, return_sequences=True)

    # 定义一个反向 LSTM 层，包含 128 个隐层，并返回完整的输出序列
    lstm_bw = LSTM(units=128, return_sequences=True, go_backwards=True)

    # 通过正向和反向 LSTM 层，从输入的特征向量序列中提取序列特征
    rnn_outputs_fw = lstm_fw(features)
    rnn_outputs_bw = lstm_bw(features)

    # 将正向和反向 LSTM 的输出连接在一起，并形成一个完整的输出序列
    rnn_outputs = concatenate([rnn_outputs_fw, rnn_outputs_bw], axis=-1)

    return rnn_outputs


# 第四步 输入FC层
def add_fc_layer(rnn_outputs):
    # 定义一个全连接层，输入 14x256 大小的 RNN 输出序列，输出 14x10 的特征向量序列
    fc_layer = TimeDistributed(Dense(units=10, activation='linear'))
    output = fc_layer(rnn_outputs)
    return output


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


def get_call_back():
    # 每隔 2 个 epoch 保存一下权重
    ckpt = ModelCheckpoint(
        filepath='/path/to/save/weights.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        period=2
    )

    # 学习率逐步降低的回调函数
    lr_schedule = LearningRateScheduler(
        lambda epoch: 1e-3 * (0.1 ** (epoch // 10))
    )

    # 网络训练过程中记录训练日志的回调函数
    class LossHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []

        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))

    history = LossHistory()

    callbacks = [ckpt, lr_schedule, history]

    return callbacks


def main(args):
    with open('config.yml') as f:
        config = dict2namespace(yaml.safe_load(f))

    set_gpu_growth()
    # 载入标注文件
    annotation_files = file_utils.get_sub_files(config.TRAIN_ANNOTATION_PATH)
    image_annotations = [load_annotation(file, config.TRAIN_IMAGE_PATH) for file in annotation_files]
    # 删除错误的图像和标注路径
    image_annotations = [ann for ann in image_annotations if os.path.exists(ann['image_path'])]
    # 加载预训练模型，训练设置参数在 config.yml 文件中
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
    grey_image = img_initialize(imagePath)

    resized_gray = cv2.resize(grey_image, (224, 224))

    # 将缩放后的灰度图像转换为 RGB 图像
    resized_rgb = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)

    # 对 RGB 图像进行归一化处理
    resized_rgb = resized_rgb.astype("float32") / 255.0

    # 将整个图像的像素值减去 ImageNet 数据集的均值
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    resized_rgb -= mean
    resized_rgb /= std

    image_input = resnet50(image_input=resized_rgb)
    feature_vector = extract_feature_vector(resized_rgb)
