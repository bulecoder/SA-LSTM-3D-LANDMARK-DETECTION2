from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models, transforms, utils
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
import os
import math
from copy import deepcopy
import pandas as pd
from MyDataLoader import Rescale, ToTensor, LandmarksDataset
import MyModel
import TrainNet
import LossFunction
import argparse

plt.ion()  # interactive mode
# Data augmentation and normalization for training


parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--landmarkNum", type=int, default=7)
# parser.add_argument("--image_scale", default=(72, 96, 96), type=tuple)
parser.add_argument("--image_scale", default=(96, 96, 96), type=tuple)

# parser.add_argument("--origin_image_size", default=(576, 768, 768), type=tuple)
parser.add_argument("--origin_image_size", default=(512, 512, 512), type=tuple)

parser.add_argument("--crop_size", default=(32, 32, 32), type=tuple)
parser.add_argument("--use_gpu", type=int, default=0)
parser.add_argument("--iteration", type=int, default=3)

parser.add_argument("--traincsv", type=str, default='train.csv')    # 下面已经定义好了数据所在的根目录，这里只需要给出具体的文件名
parser.add_argument("--testcsv", type=str, default='test.csv')
parser.add_argument("--saveName", type=str, default='test')         # 修改配置以后要修改saveName来保存训练数据
parser.add_argument("--testName", type=str, default="Full_final_64")

parser.add_argument("--R1", type=int, default=5)
parser.add_argument("--R2", type=int, default=9)

parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--data_enhanceNum", type=int, default=1)
parser.add_argument("--stage", type=str, default="train")

# 自己添加的参数    后修还可以添加 --weight_decay  --betas等
parser.add_argument('--lr', type=float, default=0.0001)


def main():
    config = parser.parse_args()
    fine_LSTM = MyModel.fine_LSTM(config).cuda(config.use_gpu)
    coarseNet = MyModel.coarseNet(config).cuda(config.use_gpu)

    # # 在测试阶段，从指定路径加载预训练好的模型权重文件，并将模型加载到指定的GPU上，map_location参数用于指定模型加载到指定的GPU上，默认为cuda(0)，即第0个GPU
    # if config.stage == 'test':
    #     fine_LSTM = torch.load('output/' + "730" + config.testName + "fine_LSTM.pkl", map_location=lambda storage, loc:storage.cuda(config.use_gpu))
    #     coarseNet = torch.load('output/' + "730" + config.testName + "coarse.pkl", map_location=lambda storage, loc:storage.cuda(config.use_gpu))

    # dataRoot = "processed_data_MICCAI/"
    dataRoot = "F:/CBCT/SA-LSTM-3D-Landmark-Detection2/processed_data/"    # 数据的根目录

    # 定义数据预处理流水线(Pipeline)包括将图像缩放到指定大小，并转换为Tensor格式
    transform_origin = transforms.Compose([
        # Rescale(config.origin_image_size),    # 图像在预处理的时候已经Resize了
        ToTensor()
    ])

    train_dataset_origin = LandmarksDataset(csv_file=dataRoot + config.traincsv,
                                            root_dir=dataRoot + "images",
                                            transform=transform_origin,
                                            landmarksNum=config.landmarkNum
                                            )

    val_dataset = LandmarksDataset(csv_file=dataRoot + config.testcsv,
                                   root_dir=dataRoot + "images",
                                   transform=transform_origin,
                                   landmarksNum=config.landmarkNum
                                   )
    
    train_dataloader = []
    val_dataloader = []

    # 创建训练数据加载器，可高效读取的批量数据
    train_dataloader_t = DataLoader(train_dataset_origin, batch_size=config.batchSize,
                                    shuffle=False, num_workers=0)
    
    if config.stage == 'train':
        for data in train_dataloader_t:
            train_dataloader.append(data)
    
    val_dataloader_t = DataLoader(val_dataset, batch_size=config.batchSize,
                                  shuffle=False, num_workers=0)
    
    for data in val_dataloader_t:
        val_dataloader.append(data)

    print(len(train_dataloader), len(val_dataloader))

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    criterion_coarse = LossFunction.coarse_heatmap(config)
    criterion_fine = LossFunction.fine_heatmap(config)

    # Observe that all parameters are being optimized
    params = list(coarseNet.parameters()) + list(fine_LSTM.parameters())

    optimizer_ft = optim.Adam(params, lr=config.lr)

    TrainNet.train_model(coarseNet, fine_LSTM, dataloaders, criterion_coarse, criterion_fine,
                         optimizer_ft, config)

if __name__ == "__main__":
    main()