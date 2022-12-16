#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   split.py
@Contact :   huahui_huang@qq.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/12     huahui_huang    1.0       生成数据划分文件
"""

import os
import random


def save_file(list, path, name):
    myfile = os.path.join(path, name)  # 全路径
    if os.path.exists(myfile):  # 移除myfile文件
        os.remove(myfile)
    with open(myfile, "w") as f:
        f.writelines(list)  # list写入myfile


def main(pic_path_folder):
    for pic_folder in os.listdir(pic_path_folder):  # 遍历目录下文件夹
        data_path = os.path.join(pic_path_folder, pic_folder)  # 文件夹路径
        num_class = len(os.listdir(data_path))  # 文件夹下文件数目
        train_list = []
        val_list = []
        train_ratio = 0.9  # 训练集所占比例
        for folder in os.listdir(data_path):  # 遍历文件夹
            if os.path.isfile(os.path.join(data_path, folder)):  # 文件路径
                continue
            train_nums = len(os.listdir(os.path.join(data_path, folder))) * train_ratio  # 训练集个数
            img_lists = os.listdir(os.path.join(data_path, folder))  # 返回包含文件名的列表
            random.shuffle(img_lists)  # 打乱顺序
            for index, img in enumerate(img_lists):  # 枚举 返回序号和文件名
                if index < train_nums:  # 分开训练集和测试集
                    train_list.append(os.path.join(data_path, folder, img) + ' ' + str(int(folder) - 1) + '\n')
                else:
                    val_list.append(os.path.join(data_path, folder, img) + ' ' + str(int(folder) - 1) + '\n')

        random.shuffle(train_list)  # 打乱顺序
        random.shuffle(val_list)
        save_file(train_list, data_path, 'train.txt')
        save_file(val_list, data_path, 'val.txt')
    print("【*】已生成数据划分文件！！！")


if __name__ == '__main__':
    pic_path_folder = os.path.join(os.getcwd(), "arthrosis")
    main(pic_path_folder)
