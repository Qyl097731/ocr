# _*_ctpn_net coding: utf-8 _*_
# @Time : 2023/5/17 23:08
# @Author : 邱依良
# @Version：V 0.1
# @File : file_utils.py
# @desc : 文件工具类

import os


def get_sub_files(path):
    """ 获取路径下所有子文件 """
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
