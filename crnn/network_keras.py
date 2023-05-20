# _*_ coding: utf-8 _*_
# @Time : 2023/5/20 21:02
# @Author : 邱依良
# @Version：V 0.1
# @File : network_keras.py
# @desc : crnn 模型

from keras.layers import (Conv2D, BatchNormalization, MaxPool2D, Input, Permute, Reshape, Dense, LeakyReLU, Activation,
                          Bidirectional, LSTM, TimeDistributed)
from keras.models import Model
from keras.layers import ZeroPadding2D
from keras.activations import relu


def keras_crnn(imgH, nc, nclass, nh, n_rnn=2, leaky_relu=False, lstm_flag=True):
    data_format = 'channels_first'  # 设置通道维靠前
    kernel_size = [3, 3, 3, 3, 3, 3, 2]  # 各卷积层卷积尺寸
    padding_size = [1, 1, 1, 1, 1, 1, 0]  # padding
    stride_size = [1, 1, 1, 1, 1, 1, 1]  # stride
    nm = [64, 128, 256, 256, 512, 512, 512]  # 卷积核个数
    image_input = Input(shape=(1, imgH, None), name='imgInput')

    def conv_relu(i, batch_normalization=False, x=None):
        # padding: one of `"valid"` or `"same"` (case-insensitive).
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        nIn = nc if i == 0 else nm[i - 1]
        nOut = nm[i]
        if leaky_relu:
            activation = LeakyReLU(alpha=0.2)
        else:
            activation = Activation(relu, name='relu{0}'.format(i))

        x = Conv2D(filters=nOut,
                   kernel_size=kernel_size[i],
                   strides=(stride_size[i], stride_size[i]),
                   padding='valid' if padding_size[i] == 0 else 'same',
                   dilation_rate=(1, 1),
                   activation=None, use_bias=True, data_format=data_format,
                   name='cnn.conv{0}'.format(i)
                   )(x)

        if batch_normalization:
            x = BatchNormalization(epsilon=1e-05, axis=1, momentum=0.1, name='cnn.batchnorm{0}'.format(i))(x)

        x = activation(x)
        return x

    x = image_input
    x = conv_relu(0, batchNormalization=False, x=x)
    x = MaxPool2D(pool_size=(2,
                             2), name='cnn.pooling{0}'.format(0), padding='valid', data_format=data_format)(
        x)
    x = conv_relu(1, batchNormalization=False, x=x)
    x = MaxPool2D(pool_size=(2,
                             2), name='cnn.pooling{0}'.format(1), padding='valid', data_format=data_format)(
        x)
    x = conv_relu(2, batchNormalization=True, x=x)
    x = conv_relu(3, batchNormalization=False, x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2,
                             2), strides=(2, 1), padding='valid', name='cnn.pooling{0}'.format(2), data_format
                  =data_format)(x)
    x = conv_relu(4, batchNormalization=True, x=x)
    x = conv_relu(5, batchNormalization=False, x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2,
                             2), strides=(2, 1), padding='valid', name='cnn.pooling{0}'.format(3), data_format
                  =data_format)(x)
    x = conv_relu(6, batchNormalization=True, x=x)
    x = Permute((3, 2, 1))(x)
    x = Reshape((-1, 512))(x)

    out = None
    if lstm_flag:
        x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True,
                               recurrent_activation='sigmoid'))(x)
        x = TimeDistributed(Dense(nh))(x)
        x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True,
                               recurrent_activation='sigmoid'))(x)
        out = TimeDistributed(Dense(nclass))(x)
    else:
        out = Dense(nclass, name='linear')(x)
    out = Reshape((-1, 1, nclass), name='out')(out)

    return Model(image_input, out)
