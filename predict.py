# _*_ coding: utf-8 _*_
# @Time : 2023/5/20 20:23
# @Author : 邱依良
# @Version：V 0.1
# @File : predict.py
# @desc : 模型预测

import os
import sys
import numpy as np
import argparse
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ctpn.utils import image_utils, np_utils, visualize
from ctpn.utils.detector import TextDetector
from ctpn.config import cur_config as config
from ctpn.layers import models


def main(args):
    # 覆盖参数
    config.USE_SIDE_REFINE = bool(args.use_side_refine)
    if args.weight_path is not None:
        config.WEIGHT_PATH = args.weight_path
    config.IMAGES_PER_GPU = 1
    config.IMAGE_SHAPE = (1024, 1024, 3)
    # 加载图片
    image, image_meta, _, _ = image_utils.load_image_gt(np.random.randint(10),
                                                        args.image_path,
                                                        config.IMAGE_SHAPE[0],
                                                        None)
    # 加载模型
    m = models.ctpn_net(config, 'test')
    m.load_weights(config.WEIGHT_PATH, by_name=True)
    # m.summary()

    # 模型预测
    text_boxes, text_scores, _ = m.predict([np.array([image]), np.array([image_meta])])
    text_boxes = np_utils.remove_pad(text_boxes[0])
    text_scores = np_utils.remove_pad(text_scores[0])[:, 0]

    # 文本行检测器
    image_meta = image_utils.parse_image_meta(image_meta)
    detector = TextDetector(config)
    text_lines = detector.detect(text_boxes, text_scores, config.IMAGE_SHAPE, image_meta['window'])
    # 可视化保存图像
    boxes_num = 30
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)
    visualize.display_polygons(image, text_lines[:boxes_num, :8], text_lines[:boxes_num, 8],
                               ax=ax)
    image_name = os.path.basename(args.image_path)
    fig.savefig('{}.{}.jpg'.format(os.path.splitext(image_name)[0], int(config.USE_SIDE_REFINE)))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_path", type=str, help="image path")
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    parse.add_argument("--use_side_refine", type=int, default=1, help="1: use side refine; 0 not use")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)
