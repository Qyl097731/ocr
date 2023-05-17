# _*_ctpn_net coding: utf-8 _*_
# @Time : 2023/5/17 23:08
# @Author : 邱依良
# @Version：V 0.1
# @File : file_utils.py
# @desc : 文件工具类

import os


def get_sub_files(directory):
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files
