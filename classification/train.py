#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train.py    
@Contact :   huahui_huang@qq.com
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/12/12     huahui_huang    1.0       None
"""
import argparse
import json
import os
from torch.cuda import amp
import torch
import torchvision
from torch import nn, device
from torch.optim import lr_scheduler
from torchsummary import summary
from torchvision.transforms import Compose, ColorJitter, Resize, ToTensor, RandomCrop, RandomRotation
from torchvision.transforms import RandomHorizontalFlip, Normalize
from tqdm import tqdm

from classification.utils.general import one_cycle
from utils.logger_cfg import get_logger
from classification.Dataset import Dataset
from classification.utils.seed_torch import seed_torch


def train(args, model, category):
    epoch = args.epoch
    logger = get_logger('train.log')
    train_txt = './arthrosis/' + category + '/train.txt'
    val_txt = './arthrosis/' + category + '/val.txt'

    train_transforms = Compose([
        Resize(size=(120, 120)),  # Compose 将用于数据集预处理的接口以列表的方式进行组合
        ColorJitter(0.3, 0.3, 0.3, 0.2),  # ColorJitter 调整图像的亮度，对比度，饱和度和色调
        RandomHorizontalFlip(),  # RandomHorizontalFlip 以一定的概率对图像进行随机水平翻转，模型训练时的数据增强操作。（默认0.5）
        RandomRotation(15),
        RandomCrop((112, 112)),  # Resize 将输入数据调整为指定大小
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transforms = Compose([
        Resize(size=(112, 112)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = Dataset(train_transforms, 'train', train_path=train_txt)  # 训练数据集，同时预处理
    val_dataset = Dataset(val_transforms, 'val', val_path=val_txt)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)  # 导入数据集
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    t_max = len(train_dataset) // args.batch_size * epoch  # 训练的上限轮数

    best_acc = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Scheduler
    if args.linear_lr:
        lf = lambda x: (1 - x / (epoch - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epoch)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    criterion = nn.CrossEntropyLoss()

    # 判断能否使用自动混合精度
    cuda = device.type != 'cpu'

    # Start training
    logger.info('{} 开始训练 ... '.format(category))
    model.train()
    for epoch in range(args.epoch):
        train_loss = 0.0
        val_loss = 0.0
        num_corrects_train = 0
        num_corrects_val = 0

        optimizer.zero_grad()
        for batch_id, data in enumerate(tqdm(train_loader)):
            with amp.autocast(enabled=cuda):
                img = data[0].cuda()
                label = data[1].cuda()
                pred = model(img)

                loss = criterion(pred, label)
                pred = pred.argmax(dim=1)
                num_corrects_train += torch.eq(pred, label).float().sum().item()
                train_loss += float(loss.item())

                if batch_id % 20 == 0:
                    logger.info("train loss:{:.4f}".format(loss.item()))

                loss.backward()
                optimizer.step()

        logger.info("train loss:{:.4f}\ttrain accuracy:{:.4f}".format(train_loss / len(train_loader),
                                                                      num_corrects_train / len(train_dataset)))

        # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------
        lr_scheduler.step()  # 调整学习率

        # 训练验证-------------------------------------------------------------------------------------------------------
        model.eval()
        for batch_id, data in enumerate(tqdm(valid_loader)):
            x_data = data[0].cuda()
            label = data[1].cuda()

            pred = model(x_data)
            loss = criterion(pred, label)
            pred = pred.argmax(dim=1)

            num_corrects_val += torch.eq(pred, label).float().sum().item()
            val_loss += float(loss.item())
        logger.info("val loss:{:.4f}\tval accuracy:{:.4f}".format(val_loss / len(valid_loader),
                                                                  num_corrects_val / len(val_dataset)))

        model.train()
        if num_corrects_val > best_acc:
            best_acc = num_corrects_val
            torch.save(model.state_dict(), './result/' + 'best_' + category + '_net.path')
            logger.info('成功保存模型')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    seed_torch()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=40, help="训练周期")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=8, help="批量数")
    parser.add_argument("--nthreads", type=str, default=8, help="线程数")
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    arthrosis = json.load(open('config/info.json', 'r'))
    for key, value in arthrosis.items():
        # efficientnet_v2_s = torchvision.models.efficientnet_v2_s(weight=EfficientNet_V2_S_Weights)
        # efficientnet_v2_s.classifier[1] = nn.Linear(1280, value[1])
        # efficientnet_v2_s.cuda()
        # summary(efficientnet_v2_s, (3, 112, 112))
        # train(args, efficientnet_v2_s, value[0])
        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50.fc = nn.Linear(2048, value[1])
        resnet50.cuda()
        summary(resnet50, (3, 112, 112))
        train(args, resnet50, value[0])
