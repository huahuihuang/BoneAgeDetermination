#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   RUS-CHN_Score.py
@Contact :   huahui_huang@qq.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/14     huahui_huang    1.0       None
"""
import argparse
import json

import cv2
from inference_utils import calc_bone_age


def main(args):
    # 参数初始化
    font = cv2.FONT_HERSHEY_DUPLEX  # 设置字体
    sex = args.sex
    label = {}
    classifer = {}
    model_dir = args.model_dir
    image_file = args.image_file
    # yolo推理-----------------------------------------------------------------------------------------------------------
    configDict = {}
    configDict['model_dir'] = model_dir
    configDict['image_file'] = image_file
    configDict['threshold'] = 0.2
    configDict['use_gpu'] = False
    configDict['run_mode'] = 'fluid'
    config = Config(configDict['model_dir'])
    detector = Detector(
        config, configDict['model_dir'], use_gpu=configDict['use_gpu'], run_mode=configDict['run_mode'])
    results = predict_image(detector, configDict['image_file'], configDict['threshold'])

    # 处理yolo推理结果----------------------------------------------------------------------------------------------------
    if len(results['boxes']) != 21:
        print("推理失败")
    for box in results['boxes']:
        if int(box[0]) not in classifer:
            classifer[int(box[0])] = []
            classifer[int(box[0])].append([int(box[2]), int(box[3]), int(box[4]), int(box[5])])
        else:
            classifer[int(box[0])].append([int(box[2]), int(box[3]), int(box[4]), int(box[5])])
    if len(classifer[0]) != 1 or len(classifer[1]) != 1 or len(classifer[2]) != 1:
        raise ValueError('推理失败')
    if len(classifer[3]) != 4 or len(classifer[4]) != 5 or len(classifer[5]) != 4 or len(classifer[6]) != 5:
        raise ValueError('推理失败')

    # True 是左手， False是右手，一般是左手
    Hand = True
    # 判断左右手，第一手指掌骨的left 比尺骨的left要大就是左手
    if classifer[2][0][0] > classifer[1][0][0]:
        Hand = True
    else:
        Hand = False

    label['Radius'] = classifer[0][0]
    label['Ulna'] = classifer[1][0]
    label['MCPFirst'] = classifer[2][0]
    # 4个MCP中，根据left的大到小排列，分出第三手指掌骨，和第五手指掌骨，因为只需要第三和第五掌骨，其他同理
    MCP = sorted(classifer[3], key=(lambda x: [x[0]]), reverse=Hand)
    label['MCPThird'] = MCP[1]
    label['MCPFifth'] = MCP[3]
    # 5个ProximalPhalanx中，根据left的大到小排列，分出第一近节指骨，第三近节指骨，第五近节指骨
    PIP = sorted(classifer[4], key=(lambda x: [x[0]]), reverse=Hand)
    label['PIPFirst'] = PIP[0]
    label['PIPThird'] = PIP[2]
    label['PIPFifth'] = PIP[4]
    # 4个MiddlePhalanx中，根据left的大到小排列，分出第三中节指骨，第三中节指骨
    MIP = sorted(classifer[5], key=(lambda x: [x[0]]), reverse=Hand)
    label['MIPThird'] = MIP[1]
    label['MIPFifth'] = MIP[3]
    # 5个DistalPhalanx中，根据left的大到小排列，分出第一远节指骨，第三远节指骨，第五远节指骨
    DIP = sorted(classifer[6], key=(lambda x: [x[0]]), reverse=Hand)
    label['DIPFirst'] = DIP[0]
    label['DIPThird'] = DIP[2]
    label['DIPFifth'] = DIP[4]

    # 每个关节通过模型推理，得到对应的等级------------------------------------------------------------------------------------
    image = cv2.imread(image_file, 0)
    results = {}
    for key, value in label.items():
        category = arthrosis[key]
        left, top, right, bottom = value
        # 从原图根据检测出来的boxes 抠出来，传入分类模型中进行预测
        image_temp = image[top:bottom, left:right]
        # 预测等级
        result = predictClass(image_temp, category[0], category[1])
        # 图上画框框和预测的等级
        # cv2.rectangle(image,(left, top), (right,bottom), (225,255,255), 2)
        # cv2.putText(image, "L:{}".format(result+1), (right+3, top +40), font, 0.9, (225,255,255), 2,)
        results[key] = result

    # 根据每个关节的等级，计算总得分-----------------------------------------------------------------------------------------
    score = 0
    SCORE = json.load(open('./config/SCORE.json', 'r'))
    for key, value in results.items():
        score += SCORE[sex][key][value]
    # 计算骨龄-----------------------------------------------------------------------------------------------------------
    boneAge = calc_bone_age(score, sex)
    # 输出报告-----------------------------------------------------------------------------------------------------------
    report = """
    第一掌骨骺分级{}级，得{}分；第三掌骨骨骺分级{}级，得{}分；第五掌骨骨骺分级{}级，得{}分；
    第一近节指骨骨骺分级{}级，得{}分；第三近节指骨骨骺分级{}级，得{}分；第五近节指骨骨骺分级{}级，得{}分；
    第三中节指骨骨骺分级{}级，得{}分；第五中节指骨骨骺分级{}级，得{}分；
    第一远节指骨骨骺分级{}级，得{}分；第三远节指骨骨骺分级{}级，得{}分；第五远节指骨骨骺分级{}级，得{}分；
    尺骨分级{}级，得{}分；桡骨骨骺分级{}级，得{}分。
    
    RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。""".format(
        results['MCPFirst'] + 1, SCORE[sex]['MCPFirst'][results['MCPFirst']],
        results['MCPThird'] + 1, SCORE[sex]['MCPThird'][results['MCPThird']],
        results['MCPFifth'] + 1, SCORE[sex]['MCPFifth'][results['MCPFifth']],
        results['PIPFirst'] + 1, SCORE[sex]['PIPFirst'][results['PIPFirst']],
        results['PIPThird'] + 1, SCORE[sex]['PIPThird'][results['PIPThird']],
        results['PIPFifth'] + 1, SCORE[sex]['PIPFifth'][results['PIPFifth']],
        results['MIPThird'] + 1, SCORE[sex]['MIPThird'][results['MIPThird']],
        results['MIPFifth'] + 1, SCORE[sex]['MIPFifth'][results['MIPFifth']],
        results['DIPFirst'] + 1, SCORE[sex]['DIPFirst'][results['DIPFirst']],
        results['DIPThird'] + 1, SCORE[sex]['DIPThird'][results['DIPThird']],
        results['DIPFifth'] + 1, SCORE[sex]['DIPFifth'][results['DIPFifth']],
        results['Ulna'] + 1, SCORE[sex]['Ulna'][results['Ulna']],
        results['Radius'] + 1, SCORE[sex]['Radius'][results['Radius']],
        score, boneAge)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dataset_group = parser.add_argument_group(title='参数设定')
    dataset_group.add_argument("--sex", type=str, default='girl', help="性别")
    dataset_group.add_argument("--model_dir", type=str, default='', help="yolo模型路径")
    dataset_group.add_argument("--image_file", type=str, default='8', help="图片路径")
    dataset_group.add_argument("--", type=str, default=8, help="线程数")
    args = parser.parse_args()

    main(args)
