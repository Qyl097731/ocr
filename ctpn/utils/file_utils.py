# _*_ctpn_net coding: utf-8 _*_
# @Time : 2023/5/17 23:08
# @Author : 邱依良
# @Version：V 0.1
# @File : file_utils.py
# @desc : 文件处理工具类

import os


def get_sub_files(dir_path, recursive=False):
    """
    获取目录下所有文件名
    :param dir_path:
    :param recursive: 是否递归
    :return:
    """
    file_paths = []
    for dir_name in os.listdir(dir_path):
        cur_dir_path = os.path.join(dir_path, dir_name)
        if os.path.isdir(cur_dir_path) and recursive:
            file_paths = file_paths + get_sub_files(cur_dir_path)
        else:
            file_paths.append(cur_dir_path)
    return file_paths
