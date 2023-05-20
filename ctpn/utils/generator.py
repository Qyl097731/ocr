# _*_ coding: utf-8 _*_
# @Time : 2023/5/20 17:46
# @Author : 邱依良
# @Version：V 0.1
# @File : generator.py
# @desc : 生成器

import numpy as np
from ..utils import image_utils, np_utils, gt_utils


def generator(image_annotations, batch_size, image_shape, width_stride,
              max_gt_num, horizontal_flip=False, random_crop=False):
    image_length = len(image_annotations)
    while True:
        ids = np.random.choice(image_length, batch_size, replace=False)
        batch_images = []
        batch_images_meta = []
        batch_gt_boxes = []
        batch_gt_class_ids = []
        for id in ids:
            image_annotation = image_annotations[id]
            image, image_meta, _, gt_quadrilaterals = image_utils.load_image_gt(id,
                                                                                image_annotation['image_path'],
                                                                                image_shape[0],
                                                                                gt_quadrilaterals=image_annotation[
                                                                                    'quadrilaterals'],
                                                                                horizontal_flip=horizontal_flip,
                                                                                random_crop=random_crop)
            class_ids = image_annotation['labels']
            gt_boxes, class_ids = gt_utils.gen_gt_from_quadrilaterals(gt_quadrilaterals,
                                                                      class_ids,
                                                                      image_shape,
                                                                      width_stride,
                                                                      box_min_size=3)
            batch_images.append(image)
            batch_images_meta.append(image_meta)
            gt_boxes = np_utils.pad_to_fixed_size(gt_boxes[:max_gt_num], max_gt_num)  # GT boxes数量防止超出阈值
            batch_gt_boxes.append(gt_boxes)
            batch_gt_class_ids.append(
                np_utils.pad_to_fixed_size(np.expand_dims(np.array(class_ids), axis=1), max_gt_num))

        # 返回结果
        yield {"input_image": np.asarray(batch_images),
               "input_image_meta": np.asarray(batch_images_meta),
               "gt_class_ids": np.asarray(batch_gt_class_ids),
               "gt_boxes": np.asarray(batch_gt_boxes)}, None
