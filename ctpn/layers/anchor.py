# _*_ coding: utf-8 _*_
# @Time : 2023/5/20 18:06
# @Author : 邱依良
# @Version：V 0.1
# @File : anchor.py.py
# @desc : ctpn anchor层，在输入图像边框外的anchors丢弃

import keras
import tensorflow as tf
import numpy as np


def generate_anchors(heights, width):
    """
    生成基准anchors
    :param heights: 高度列表
    :param width: 宽度，数值
    :return:
    """
    w = np.array([width] * len(heights))
    h = np.array(heights)
    return np.stack([-0.5 * h, -0.5 * w, 0.5 * h, 0.5 * w], axis=1)


def shift(shape, stride, base_anchors):
    """
    根据feature map的长宽，生成所有的anchors
    :param shape: （H,W)
    :param stride: 步长
    :param base_anchors:所有的基准anchors，(anchor_num,4)
    :return:
    """
    H, W = shape[0], shape[1]
    ctr_x = (tf.cast(tf.range(W), tf.float32) + tf.constant(0.5, dtype=tf.float32)) * stride
    ctr_y = (tf.cast(tf.range(H), tf.float32) + tf.constant(0.5, dtype=tf.float32)) * stride

    ctr_x, ctr_y = tf.meshgrid(ctr_x, ctr_y)

    # 打平为1维,得到所有锚点的坐标
    ctr_x = tf.reshape(ctr_x, [-1])
    ctr_y = tf.reshape(ctr_y, [-1])
    #  (H*W,1,4)
    shifts = tf.expand_dims(tf.stack([ctr_y, ctr_x, ctr_y, ctr_x], axis=1), axis=1)
    # (1,anchor_num,4)
    base_anchors = tf.expand_dims(tf.constant(base_anchors, dtype=tf.float32), axis=0)

    # (H*W,anchor_num,4)
    anchors = shifts + base_anchors
    # 转为(H*W*anchor_num,4) 返回
    return tf.reshape(anchors, [-1, 4])


def filter_out_of_bound_boxes(boxes, feature_shape, stride):
    """
    过滤图像边框外的anchor
    :param boxes: [n,y1,x1,y2,x2]
    :param feature_shape: 特征图的长宽 [h,w]
    :param stride: 网络步长
    :return:
    """
    # 图像原始长宽为特征图长宽*步长
    h, w = feature_shape[0], feature_shape[1]
    h = tf.cast(h * stride, tf.float32)
    w = tf.cast(w * stride, tf.float32)

    valid_boxes_tag = tf.logical_and(tf.logical_and(tf.logical_and(boxes[:, 0] >= 0,
                                                                   boxes[:, 1] >= 0),
                                                    boxes[:, 2] <= h),
                                     boxes[:, 3] <= w)
    boxes = tf.boolean_mask(boxes, valid_boxes_tag)
    valid_boxes_indices = tf.where(valid_boxes_tag)[:, 0]
    return boxes, valid_boxes_indices


class CtpnAnchor(keras.layers.Layer):
    def __init__(self, heights, width, stride, **kwargs):
        """
        :param heights: 高度列表
        :param width: 宽度，数值，如：16
        :param stride: 步长,
        :param image_shape: tuple(H,W,C)
        """
        self.heights = heights
        self.width = width
        self.stride = stride
        # base anchors数量
        self.num_anchors = None  # 初始化值
        super(CtpnAnchor, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs：输入 卷积层特征(锚点所在层)，shape：[batch_size,H,W,C]
        :param kwargs:
        :return:
        """
        features = inputs
        features_shape = tf.shape(features)
        print("feature_shape:{}".format(features_shape))

        base_anchors = generate_anchors(self.heights, self.width)
        # print("len(base_anchors):".format(len(base_anchors)))
        anchors = shift(features_shape[1:3], self.stride, base_anchors)
        anchors, valid_anchors_indices = filter_out_of_bound_boxes(anchors, features_shape[1:3], self.stride)
        self.num_anchors = tf.shape(anchors)[0]
        # 扩展第一维，batch_size;每个样本都有相同的anchors
        anchors = tf.tile(tf.expand_dims(anchors, axis=0), [features_shape[0], 1, 1])
        valid_anchors_indices = tf.tile(tf.expand_dims(valid_anchors_indices, axis=0), [features_shape[0], 1])

        return [anchors, valid_anchors_indices]

    def compute_output_shape(self, input_shape):
        """

        :param input_shape: [batch_size,H,W,C]
        :return:
        """
        # 计算所有的anchors数量
        total = self.num_anchors
        return [(input_shape[0], total, 4),
                (input_shape[0], total)]


def main():
    anchors = generate_anchors([11, 16, 23, 33, 48, 68, 97, 139, 198, 283], 16)
    print(anchors)
