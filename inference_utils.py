#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   inference_utils.py
@Contact :   huahui_huang@qq.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/14     huahui_huang    1.0       None
"""

import math
import os

import numpy as np
import paddle
import paddle.fluid as fluid
import yaml


def calc_bone_age(score, sex):
    """
    根据小关节总分和性别计算骨龄
    :param score: 小关节总分
    :param sex: 性别
    :return: 骨龄
    """
    if sex == 'boy':
        boneAge = 2.01790023656577 + (-0.0931820870747269) * score + math.pow(score, 2) * 0.00334709095418796 + \
                  math.pow(score, 3) * (-3.32988302362153E-05) + math.pow(score, 4) * 1.75712910819776E-07 + \
                  math.pow(score, 5) * (-5.59998691223273E-10) + math.pow(score, 6) * 1.1296711294933E-12 + \
                  math.pow(score, 7) * (-1.45218037113138e-15) + math.pow(score, 8) * 1.15333377080353e-18 + \
                  math.pow(score, 9) * (-5.15887481551927e-22) + math.pow(score, 10) * 9.94098428102335e-26
        return round(boneAge, 2)
    elif sex == 'girl':
        boneAge = 5.81191794824917 + (-0.271546561737745) * score + \
                  math.pow(score, 2) * 0.00526301486340724 + math.pow(score, 3) * (-4.37797717401925E-05) + \
                  math.pow(score, 4) * 2.0858722025667E-07 + math.pow(score, 5) * (-6.21879866563429E-10) + \
                  math.pow(score, 6) * 1.19909931745368E-12 + math.pow(score, 7) * (-1.49462900826936E-15) + \
                  math.pow(score, 8) * 1.162435538672E-18 + math.pow(score, 9) * (-5.12713017846218E-22) + \
                  math.pow(score, 10) * 9.78989966891478E-26
        return round(boneAge, 2)


class Detector(object):

    def __init__(self,
                 config,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 threshold=0.5):
        self.config = config
        self.predictor = load_predictor(
            model_dir,
            run_mode=run_mode,
            min_subgraph_size=self.config.min_subgraph_size,
            use_gpu=use_gpu)

    def preprocess(self, im):
        preprocess_ops = []
        for op_info in self.config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            if op_type == 'Resize':
                new_op_info['arch'] = self.config.arch
            preprocess_ops.append(eval(op_type)(**new_op_info))
        im, im_info = preprocess(im, preprocess_ops)
        inputs = create_inputs(im, im_info, self.config.arch)
        return inputs, im_info

    def postprocess(self, np_boxes, np_masks, np_lmk, im_info, threshold=0.5):
        results = {}
        expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        results['boxes'] = np_boxes
        return results

    def predict(self,
                image,
                threshold=0.2,
                warmup=0,
                repeats=1,
                run_benchmark=False):

        inputs, im_info = self.preprocess(image)
        np_boxes, np_masks, np_lmk = None, None, None

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_tensor(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(repeats):
            self.predictor.zero_copy_run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_tensor(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()

        results = []
        if not run_benchmark:
            results = self.postprocess(
                np_boxes, np_masks, np_lmk, im_info, threshold=threshold)
        return results


def create_inputs(im, im_info, model_arch='YOLO'):
    inputs = {}
    inputs['image'] = im
    origin_shape = list(im_info['origin_shape'])
    resize_shape = list(im_info['resize_shape'])
    pad_shape = list(im_info['pad_shape']) if im_info[
                                                  'pad_shape'] is not None else list(im_info['resize_shape'])
    scale_x, scale_y = im_info['scale']
    im_size = np.array([origin_shape]).astype('int32')
    inputs['im_size'] = im_size
    return inputs


def load_predictor(model_dir,
                   run_mode='fluid',
                   batch_size=1,
                   use_gpu=False,
                   min_subgraph_size=3):
    if not use_gpu and not run_mode == 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
            .format(run_mode, use_gpu))
    if run_mode == 'trt_int8':
        raise ValueError("TensorRT int8 mode is not supported now, "
                         "please use trt_fp32 or trt_fp16 instead.")
    precision_map = {
        'trt_int8': fluid.core.AnalysisConfig.Precision.Int8,
        'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
        'trt_fp16': fluid.core.AnalysisConfig.Precision.Half
    }
    config = fluid.core.AnalysisConfig(
        os.path.join(model_dir, '__model__'),
        os.path.join(model_dir, '__params__'))
    if use_gpu:
        config.enable_use_gpu(100, 0)
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()

    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 10,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=False)

    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(config)
    return predictor


def predict_image(detector, image_file, threshold):
    results = detector.predict(image_file, threshold)
    return results


class Config:
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.use_python_inference = yml_conf['use_python_inference']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']


def predict_class(im, classifer, num_classes):
    """
    预测小关节的等级
    :param im: 小关节图片
    :param classifer: 小关节的名称
    :param num_classes: 小关节的最大等级
    :return: 小关节的具体等级
    """
    from paddle.vision.transforms import Compose, Resize, Normalize, Transpose
    transforms = Compose([Resize(size=(224, 224)),
                          Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format='HWC'),
                          Transpose()])
    model = paddle.vision.models.resnet50(num_classes=num_classes)
    model_path = '/home/aistudio/work/out2/best_' + classifer + '_net.pdparams'
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()
    im = np.expand_dims(im, 2)
    infer_data = transforms(im)
    infer_data = np.expand_dims(infer_data, 0)
    infer_data = paddle.to_tensor(infer_data, dtype='float32')
    result = model(infer_data)[0]  # 关键代码，实现预测功能
    result = np.argmax(result.numpy())  # 获得最大值所在的序号
    return result
