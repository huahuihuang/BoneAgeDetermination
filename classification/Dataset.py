#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Dataset.py    
@Contact :   huahui_huang@qq.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/12     huahui_huang    1.0       None
"""

import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 transforms,  # 图像的transform
                 mode='train',  # 训练模式，train、val和test
                 train_path=None,  # 训练列表文件路径，文件中每一行第一个是样本文件，第二个是标注。
                 val_path=None,  # 验证列表文件路径，与训练文件一致。
                 separator=' ', ):  # 指定列表文件中样本文件和训练文件的分隔符，默认是空格
        # 构建数据增强对象
        self.transforms = transforms
        # 新建一个保存文件路径的空列表
        self.file_list = list()
        # 将模式类型字符串转换为小写并保存为成员变量
        self.mode = mode.lower()
        # 各个类型的文件列表保存到file_path变量中。
        if self.mode == 'train':
            file_path = train_path
        elif self.mode == 'val':
            file_path = val_path
        # 打开列表文件，文件包含若干行，数量与数据集样本数量相同，训练集(train)和验证集(val)列表包含样本路径和标签。
        with open(file_path, 'r') as f:
            for line in f:
                # 分离样本路径和标签。
                items = line.strip().split(separator)
                image_path = items[0]
                label = int(items[1])
                # 将样本路径和标签保存在列表中。
                self.file_list.append([image_path, label])

    def __getitem__(self, idx):  # 用于按照索引读取每个元素的具体内容
        # 通过idx下标，在file_list里获取样本图片路径和标签。
        image_path, label = self.file_list[idx]
        im = Image.open(image_path).convert('RGB')
        im = self.transforms(im)
        return im, label

    def __len__(self):  # 返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.file_list)
