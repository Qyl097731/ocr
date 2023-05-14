# _*_ coding: utf-8 _*_
# @Time : 2023/5/14 20:11
# @Author : 邱依良
# @Version：V 0.1
# @File : Config.py
# @desc : 配置类

class Config:
    """模型配置参数类"""
    # 模型输入参数
    input_size = (224, 244)
    num_classes = 10

    # 训练参数
    batch_size = 32
    learning_rate = 0.01
    epochs = 100

    anchors_height = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]

    # 其他配置参数
    use_side_refine = True
    show_progress = False
