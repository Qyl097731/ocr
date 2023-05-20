# _*_ coding: utf-8 _*_
# @Time : 2023/5/20 21:26
# @Author : 邱依良
# @Version：V 0.1
# @File : utils.py
# @desc : 工具类


from PIL import Image
import numpy as np


class StrLabelConverter(object):

    def __init__(self, alphabet):
        self.alphabet = alphabet + 'ç'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, res):
        N = len(res)
        raw = []
        for i in range(N):
            if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
                raw.append(self.alphabet[res[i] - 1])
        return ''.join(raw)


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        size = self.size
        imgW, imgH = size
        scale = img.size[1] * 1.0 / imgH
        w = img.size[0] / scale
        w = int(w)
        img = img.resize((w, imgH), self.interpolation)
        w, h = img.size
        img = (np.array(img) / 255.0 - 0.5) / 0.5

        return img
