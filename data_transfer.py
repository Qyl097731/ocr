# _*_ coding: utf-8 _*_
# @Time : 2023/5/14 1:12
# @Author : 邱依良
# @Version：V 0.1
# @File : data_transfer.py
# @desc : 将gnt转换成png、dgrl转换成png

import os
import struct
import cv2 as cv
import numpy as np
from PIL import Image

character_test_data_original_dir = './data/character/test/gnt/'
character_test_data_target_dir = './data/character/test/png/'

character_train_data_original_dir = './data/character/train/gnt/'
character_train_data_target_dir = './data/character/train/png/'

textline_test_data_original_dir = './data/textline/test/dgrl/'
textline_test_data_target_dir = './data/textline/test/png/'

textline_train_data_original_dir = './data/textline/train/dgrl/'
textline_train_data_target_dir = './data/textline/train/png/'


def transfer_gnt_to_png(original_dir, target_dir, charset):
    dirs = os.listdir(original_dir)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    num = 0
    for dir in dirs:
        for filename in os.listdir(original_dir + dir):
            file = dir + '/' + filename
            tag = []
            img_bytes = []
            img_wid = []
            img_hei = []
            f = open(original_dir + file, "rb")
            while f.read(4):
                tag_code = f.read(2)
                tag.append(tag_code)
                width = struct.unpack('<h', bytes(f.read(2)))
                height = struct.unpack('<h', bytes(f.read(2)))
                img_hei.append(height[0])
                img_wid.append(width[0])
                data = f.read(width[0] * height[0])
                img_bytes.append(data)
            f.close()
            for k in range(0, len(tag)):
                im = Image.frombytes('L', (img_wid[k], img_hei[k]), img_bytes[k])
                if os.path.exists(target_dir + tag[k].decode(charset)):
                    im.save(target_dir + tag[k].decode(charset) + "/" + str(num) + ".jpg")
                else:
                    os.mkdir(target_dir + tag[k].decode(charset))
                    im.save(target_dir + tag[k].decode(charset) + "/" + str(num) + ".jpg")
            num = num + 1
        print(tag.__len__())

        files = os.listdir(target_dir)
        n = 0
        f = open("label.txt", "w")  # 创建用于训练的标签文件
        for file in files:
            files_d = os.listdir(target_dir + file)
            for file1 in files_d:
                f.write(file + "/" + file1 + " " + str(n) + "\n")
            n = n + 1


def read_from_dgrl(original_dir):
    if not os.path.exists(original_dir):
        print('DGRL not exis!')
        return

    dir_name, base_name = os.path.split(original_dir)
    label_dir = dir_name + '_label'
    image_dir = dir_name + '_images'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    with open(original_dir, 'rb') as f:
        # 读取表头尺寸
        header_size = np.fromfile(f, dtype='uint8', count=4)
        header_size = sum([j << (i * 8) for i, j in enumerate(header_size)])
        # print(header_size)

        # 读取表头剩下内容，提取 code_length
        header = np.fromfile(f, dtype='uint8', count=header_size - 4)
        code_length = sum([j << (i * 8) for i, j in enumerate(header[-4:-2])])
        # print(code_length)

        # 读取图像尺寸信息，提取图像中行数量
        image_record = np.fromfile(f, dtype='uint8', count=12)
        height = sum([j << (i * 8) for i, j in enumerate(image_record[:4])])
        width = sum([j << (i * 8) for i, j in enumerate(image_record[4:8])])
        line_num = sum([j << (i * 8) for i, j in enumerate(image_record[8:])])
        print('图像尺寸:')
        print(height, width, line_num)

        # 读取每一行的信息
        for k in range(line_num):
            print(k + 1)

            # 读取该行的字符数量
            char_num = np.fromfile(f, dtype='uint8', count=4)
            char_num = sum([j << (i * 8) for i, j in enumerate(char_num)])
            print('字符数量:', char_num)

            # 读取该行的标注信息
            label = np.fromfile(f, dtype='uint8', count=code_length * char_num)
            label = [label[i] << (8 * (i % code_length))
                     for i in range(code_length * char_num)]
            label = [sum(label[i * code_length:(i + 1) * code_length])
                     for i in range(char_num)]
            label = [struct.pack('I', i).decode('gbk', 'ignore')[0] for i in label]
            print('合并前：', label)
            label = ''.join(label)
            # 去掉不可见字符 \x00，这一步不加的话后面保存的内容会出现看不见的问题
            label = ''.join(label.split(b'\x00'.decode()))
            print('合并后：', label)

            # 读取该行的位置和尺寸
            pos_size = np.fromfile(f, dtype='uint8', count=16)
            y = sum([j << (i * 8) for i, j in enumerate(pos_size[:4])])
            x = sum([j << (i * 8) for i, j in enumerate(pos_size[4:8])])
            h = sum([j << (i * 8) for i, j in enumerate(pos_size[8:12])])
            w = sum([j << (i * 8) for i, j in enumerate(pos_size[12:])])
            # print(x, y, w, h)

            # 读取该行的图片
            bitmap = np.fromfile(f, dtype='uint8', count=h * w)
            bitmap = np.array(bitmap).reshape(h, w)

            # 保存信息
            label_file = os.path.join(
                label_dir, base_name.replace('.dgrl', '_' + str(k) + '.txt'))
            with open(label_file, 'w') as f1:
                f1.write(label)
            bitmap_file = os.path.join(
                image_dir, base_name.replace('.dgrl', '_' + str(k) + '.jpg'))
            cv.imwrite(bitmap_file, bitmap)


if __name__ == '__main__':
    # transfer_gnt_to_png(character_train_data_original_dir, character_train_data_target_dir, 'gbk')

    # transfer_gnt_to_png(character_test_data_original_dir, character_test_data_target_dir, 'gbk')

    transfer_dgrl_to_png()