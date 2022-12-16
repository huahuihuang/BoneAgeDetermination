#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   logger_cfg.py    
@Contact :   huahui_huang@qq.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/15     huahui_huang    1.0       日志输出工具
"""

import logging
import os
import sys
from logging import handlers


def logger_train():
    os.makedirs('./run/train', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',  # 日志输出格式
                        datefmt='%a, %d %b %Y %H:%M:%S',  # 定义日期格式
                        filename='./run/train/train.log',  # 日志输出路径
                        filemode='w'  # 覆盖掉旧日志 ，默认时追加
                        )
    logger = logging.getLogger(__name__)
    return logger


dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
LOG_ROOT = dirname


def get_logger(log_filename, level=logging.INFO, when='D', back_count=0):
    """
    :brief  日志记录
    :param log_filename: 日志名称
    :param level: 日志等级
    :param when: 间隔时间:
        S:秒
        M:分
        H:小时
        D:天
        W:每星期（interval==0时代表星期一）
        midnight: 每天凌晨
    :param back_count: 备份文件的个数，若超过该值，就会自动删除
    :return: logger
    """
    # 创建一个日志器。提供了应用程序接口
    logger = logging.getLogger(log_filename)
    # 设置日志输出的最低等级,低于当前等级则会被忽略
    logger.setLevel(level)
    # 创建日志输出路径
    log_path = os.path.join(LOG_ROOT, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file_path = os.path.join(log_path, log_filename)
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    # 创建处理器：ch为控制台处理器，fh为文件处理器
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # 输出到文件
    fh = logging.handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        when=when,
        backupCount=back_count,
        encoding='utf-8')
    fh.setLevel(level)
    # 设置日志输出格式
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 将处理器，添加至日志器中
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
