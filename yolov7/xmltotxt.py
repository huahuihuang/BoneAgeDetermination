#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   xmltotxt.py    
@Contact :   huahui_huang@qq.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/15     huahui_huang    1.0       None
"""

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2 as cv


def main(mode='train'):
    # ******改成自己的类别名称***********
    classes = ['Radius', 'Ulna', 'MCPFirst', 'ProximalPhalanx', 'MCP', 'DistalPhalanx', 'MiddlePhalanx']

    xml_file_path = f'./HandBoneAgeX_ray/{mode}/xml_labels/'  # *********检查和自己的xml文件夹名称是否一致*********
    images_file_path = f'./HandBoneAgeX_ray/{mode}/images/'  # *******检查和自己的图像文件夹名称是否一致********

    # 此处不要改动，只是创一个临时文件夹
    if not os.path.exists(f'./HandBoneAgeX_ray/{mode}/labels/'):
        os.makedirs(f'./HandBoneAgeX_ray/{mode}/labels/')
    txt_file_path = f'./HandBoneAgeX_ray/{mode}/labels/'

    def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def convert_annotations(image_name):
        in_file = open(xml_file_path + image_name + '.xml', encoding='UTF-8')
        out_file = open(txt_file_path + image_name + '.txt', 'w', encoding='UTF-8')
        tree = ET.parse(in_file)
        root = tree.getroot()

        # 读取图片
        src = cv.imread(images_file_path + image_name + '.png')
        w = int(src.shape[1])
        h = int(src.shape[0])
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            cls = obj.find('name').text
            # if cls not in classes or int(difficult) == 1:
            #     continue
            if cls not in classes == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    total_xml = os.listdir(xml_file_path)
    num_xml = len(total_xml)  # XML文件总数

    for i in tqdm(range(num_xml)):
        name = total_xml[i][:-4]
        convert_annotations(name)


if __name__ == '__main__':
    main(mode='train')
    main(mode='val')
