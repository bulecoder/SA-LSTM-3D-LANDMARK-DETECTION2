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
from skimage import io, transform
import math
from copy import deepcopy
import pandas as pd
import math
import copy
import time
import PIL
# import angle
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom
import MyUtils
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar):

    #print (recon_x.view(-1))
    #BCE = F.binary_cross_entropy(recon_x.view(1,-1), x.view(1,-1), size_average = False)
    # BCE = F.mse_loss(recon_x.view(1,-1), x.view(1,-1), size_average = False)
    BCE = F.l1_loss(recon_x.view(1,-1), x.view(1,-1), size_average = False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # return BCE + KLD
    return BCE


class attentionLoss(nn.Module):
    def __init__(self, gpu):
        super(attentionLoss, self).__init__()
        self.lossFun = torch.nn.L1Loss(size_average=False)
        self.gpu = gpu
        self.delta = torch.tensor([0.]).cuda(self.gpu)

    def forward(self, coormeanAngles, labelsAngles, attention):
        topN = coormeanAngles.size()[0]
        topkP, indexs = torch.topk(attention, topN)
        indexs = indexs.cpu().numpy()
        loss = 0
        alll = 0
        for i in range(topN):
            temp = attention[indexs[i]]
            alll += temp
            # ~ temp = 1
            loss += temp * self.lossFun(coormeanAngles[i, :], labelsAngles[i, :])
        # ~ loss += self.lossFun(1/(torch.var(attention)*1e8 + 1), self.delta)
        # ~ print (alll/topN - 1/ 5456)
        # ~ print (torch.var(attention))
        return loss

# TODO: 用到了这个
class coarse_heatmap(nn.Module):
    def __init__(self, config):
        # use_gpu, batchSize, landmarkNum, image_scale
        super(coarse_heatmap, self).__init__()
        self.use_gpu = config.use_gpu
        self.batchSize = config.batchSize
        self.landmarkNum = config.landmarkNum
        self.l1Loss = nn.L1Loss(reduction='none') # 改为 none 以便应用 Mask
        self.Long, self.higth, self.width = config.image_scale # (128, 128, 128)
        
        # 创建一个足够大的高斯热图模板 (2倍尺寸)
        self.HeatMap_groundTruth = torch.zeros(self.Long * 2, self.higth * 2, self.width * 2).cuda(self.use_gpu)

        rr = 21  # 半径
        dev = 2  # 标准差
        referPoint = (self.Long, self.higth, self.width)  # 中心点
        
        # 预计算高斯分布
        # 优化：可以使用网格生成避免三重循环，但只运行一次初始化，忍了
        for k in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for i in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                for j in range(referPoint[2] - rr, referPoint[2] + rr + 1):
                    temdis = MyUtils.Mydist3D(referPoint, (k, i, j))
                    if temdis <= rr:
                        self.HeatMap_groundTruth[k][i][j] = math.exp(-1 * temdis**2 / (2 * dev**2))

    def forward(self, predicted_heatmap, local_coordinate, labels, phase):
        loss = 0
        total_valid_points = 0
        
        # labels shape: (B, N, 3) 归一化坐标 (0~1) 或者 -1 (缺失)
        
        # 将归一化坐标转换为像素坐标
        # x, y, z 分别对应 High, Width, Long (注意这里的顺序需要和 MyDataLoader 一致)
        # 根据 MyDataLoader，输入是 (128, 128, 128)，所以乘的系数一样
        
        # labels_pixel shape: (B, N, 3)
        scale_tensor = torch.tensor([self.higth - 1, self.width - 1, self.Long - 1]).cuda(self.use_gpu)
        labels_pixel = labels * scale_tensor
        labels_pixel = torch.round(labels_pixel).long() # 转整数索引

        batch_size = labels.shape[0]

        for b in range(batch_size):
            # 获取该样本的所有点坐标
            X = labels_pixel[b, :, 0] # H
            Y = labels_pixel[b, :, 1] # W
            Z = labels_pixel[b, :, 2] # D
            
            # 获取原始归一化标签用于判断 Mask
            raw_labels = labels[b]

            for i in range(self.landmarkNum):
                # Mask check: 如果坐标是负数 (缺失值)，跳过
                if raw_labels[i, 0] < 0:
                    continue
                
                # 防止越界 (虽然 MyDataLoader 里处理了，但 safe check 很重要)
                z_idx = torch.clamp(Z[i], 0, self.Long - 1)
                x_idx = torch.clamp(X[i], 0, self.higth - 1)
                y_idx = torch.clamp(Y[i], 0, self.width - 1)

                # 根据真实位置裁剪热力图 GT
                # 逻辑：从 HeatMap_groundTruth 中心扣出一块和预测图一样大的
                # 如果点在左上角 (0,0,0)，就取 HeatMap 右下部分
                # 索引逻辑比较绕，沿用原作者思路但增加安全性
                
                # 原始逻辑：self.Long - Z[i]
                start_z = self.Long - z_idx
                start_x = self.higth - x_idx
                start_y = self.width - y_idx
                
                coarse_heatmap_gt = self.HeatMap_groundTruth[
                    start_z : start_z + self.Long,
                    start_x : start_x + self.higth,
                    start_y : start_y + self.width
                ]
                
                # 归一化 GT
                if coarse_heatmap_gt.sum() > 0:
                    coarse_heatmap_gt = coarse_heatmap_gt / coarse_heatmap_gt.sum()

                # 计算 L1 Loss
                # predicted_heatmap shape: (B, N, D, H, W)
                pred = predicted_heatmap[i][b]
                
                loss += torch.abs(pred - coarse_heatmap_gt).sum()
                total_valid_points += 1
        
        if total_valid_points > 0:
            return loss / total_valid_points
        else:
            return torch.tensor(0.0).cuda(self.use_gpu)

# TODO： 用到了这个
class fine_heatmap(nn.Module):
    def __init__(self, config):
        # use_gpu, batchSize, landmarkNum, cropSize
        super(fine_heatmap, self).__init__()
        self.use_gpu = config.use_gpu
        self.batchSize = config.batchSize
        self.landmarkNum = config.landmarkNum
        
        self.Long, self.higth, self.width = config.crop_size # (64, 64, 64) or similar
        self.binaryLoss = nn.BCEWithLogitsLoss(reduction='none') # 改为 none

        self.HeatMap_groundTruth = torch.zeros(self.Long, self.higth, self.width).cuda(self.use_gpu)

        rr = 11
        dev = 2
        referPoint = (self.Long//2, self.higth//2, self.width//2)
        
        # 预计算高斯分布 (Center-based)
        for k in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for i in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                for j in range(referPoint[2] - rr, referPoint[2] + rr + 1):
                    temdis = MyUtils.Mydist3D(referPoint, (k, i, j))
                    if temdis <= rr:
                        self.HeatMap_groundTruth[k][i][j] = math.exp(-1 * temdis**2 / (2 * dev**2))

    def forward(self, predicted_heatmap, labels=None):
        # 注意：原代码 forward 只接受 predicted_heatmap
        # 我们修改了 TrainNet 调用，传入 labels 以便做 Mask
        
        # 如果 TrainNet 没传 labels，就假设全部有效 (兼容旧代码)
        if labels is None:
            mask = torch.ones(predicted_heatmap.shape[0], self.landmarkNum).cuda(self.use_gpu)
        else:
            # labels: (B, N, 3) -> mask: (B, N)
            mask = (labels[:, :, 0] >= 0).float()

        loss = 0
        total_valid = 0

        # predicted_heatmap: (B, N, D, H, W)
        batch_size = predicted_heatmap.shape[0]

        for b in range(batch_size):
            for i in range(self.landmarkNum):
                # 如果该点缺失，跳过
                if mask[b, i] == 0:
                    continue
                
                # 计算 BCE Loss
                # 这里的 GT 是以中心为峰值的高斯分布 (因为 Fine Stage 是 Offset 回归，目标就是中心)
                # 或者是 Heatmap Regression on cropped patch?
                # SA-LSTM 的 fine stage 是回归相对于 patch 中心的 heatmap
                
                bce = self.binaryLoss(predicted_heatmap[b, i], self.HeatMap_groundTruth)
                loss += bce.sum()
                total_valid += 1
        
        if total_valid > 0:
            return loss / total_valid # 平均 Loss
        else:
            return torch.tensor(0.0).cuda(self.use_gpu)

class fine_offset(nn.Module):
    def __init__(self, use_gpu, batchSize, landmarkNum):
        super(fine_offset, self).__init__()
        self.use_gpu = use_gpu
        self.batchSize = batchSize
        self.landmarkNum = landmarkNum
        self.l1Loss = nn.L1Loss(size_average=False)

    def forward(self, predicted_coordinate, local_coordinate, labels, ROIs_b, base_coordinate, phase):
        loss = 0
        # print('predict ', predicted_point_offsets[0])
        # print(labels[2,:])
        # print('gt ', torch.mean(predicted_point_offsets[2]+ point_coordinate[2], dim=0))

        # for i in range(self.landmarkNum):
        #     repeat_labels = labels[i, :].unsqueeze(0).repeat(predicted_point_offsets[i].size()[0], 1)
        #     loss += self.l1Loss(predicted_point_offsets[i], repeat_labels)
            # print(point_coordinate[i])
        # return loss
        for i in range(self.landmarkNum):
            # repeat_labels = labels[0, i, :].repeat(32, 32, 32, 1)
            # print(repeat_labels.size())
            # print(labels[0, i, :])
            # print(repeat_labels[3, 3, 1, :])
            # print(repeat_labels[15, 10, 2, :])


            # repeat_ROIs = ROIs[i, :].unsqueeze(0).repeat(predicted_point_offsets[i].size()[0], 1)
            # # print("repeat_labels", repeat_labels.size())
            # # print(labels[i, :].unsqueeze(0).detach().cpu().numpy() * np.array([767, 767, 575]))
            if phase == 'val':
            #     print(labels)
            #     print((torch.abs(local_coordinate[i] - labels[0, i, :]) - torch.abs(predicted_coordinate[i] - labels[0, i, :])).view(-1, 3)[0:30, :].detach().cpu().numpy() * np.array([767, 767, 575]))
            #     print((local_coordinate[i] - labels[0, i, :]).view(-1, 3)[0:30, :].detach().cpu().numpy() * np.array([767, 767, 575]))
                print()
                print((labels[0, i, :] - local_coordinate[i]).view(-1, 3)[320:352, :].detach().cpu().numpy() * np.array([767, 767, 575]))
                print((predicted_coordinate[i] - local_coordinate[i]).view(-1, 3)[320:352, :].detach().cpu().numpy() * np.array([767, 767, 575]))
                print()
                # print((local_coordinate[i])1.view(-1, 3)[320:352, :].detach().cpu().numpy() * np.array([767, 767, 575]))
            # if phase == 'val':
            #     print((torch.abs(predicted_point_offsets[i] - repeat_labels))[0:30, :].detach().cpu().numpy() * np.array([767, 767, 575]))

            loss += torch.abs(predicted_coordinate[i] - labels[0, i, :]).sum()
            # # print(point_coordinate[i])
        return loss

class fusionLossFunc_improved(nn.Module):
    def __init__(self, use_gpu, R, imageSize, imageNum, landmarkNum):
        super(fusionLossFunc_improved, self).__init__()

        l, h, w = 72, 96, 96
        # ~ l, h, w = 144, 192, 192

        self.use_gpu = use_gpu
        self.R = R
        self.width = w
        self.higth = h
        self.Long = l

        self.binaryLoss = nn.BCEWithLogitsLoss(size_average=False)
        # ~ self.binaryLoss = nn.BCEWithLogitsLoss()

        self.huberLoss = torch.nn.L1Loss()
        # ~ self.offsetMask = torch.zeros(h, w).cuda(self.use_gpu)

        self.offsetMapx = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapy = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapz = np.ones((self.Long * 2, self.higth * 2, self.width * 2))

        self.HeatMap = np.zeros((self.Long * 2, self.higth * 2, self.width * 2))

        self.binary_class_groundTruth = Variable(torch.zeros(imageNum, landmarkNum, l, h, w).cuda(self.use_gpu))

        rr = R
        referPoint = (self.Long, self.higth, self.width)
        for k in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for i in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                for j in range(referPoint[2] - rr, referPoint[2] + rr + 1):
                    temdis = MyUtils.Mydist3D(referPoint, (k, i, j))
                    if temdis <= rr:
                        self.HeatMap[k][i][j] = 1

        for i in range(2 * self.Long):
            self.offsetMapz[i, :, :] = self.offsetMapz[i, :, :] * i

        for i in range(2 * self.higth):
            self.offsetMapx[:, i, :] = self.offsetMapx[:, i, :] * i

        for i in range(2 * self.width):
            self.offsetMapy[:, :, i] = self.offsetMapy[:, :, i] * i

        self.offsetMapz = referPoint[0] - self.offsetMapz
        self.offsetMapx = referPoint[1] - self.offsetMapx
        self.offsetMapy = referPoint[2] - self.offsetMapy

        # print (self.HeatMap)
        # print (self.offsetMapx)
        # print (self.offsetMapy)

        self.HeatMap = Variable(torch.from_numpy(self.HeatMap)).cuda(self.use_gpu).float()

        self.offsetMapx = Variable(torch.from_numpy(self.offsetMapx)).cuda(self.use_gpu).float() / rr
        self.offsetMapy = Variable(torch.from_numpy(self.offsetMapy)).cuda(self.use_gpu).float() / rr
        self.offsetMapz = Variable(torch.from_numpy(self.offsetMapz)).cuda(self.use_gpu).float() / rr

        return

    def forward(self, featureMaps, landmarks):
        # ~ print (featureMaps.size())

        imageNum = featureMaps[0].size()[0]
        # ~ landmarkNum = int(featureMaps[0].size()[1]/3)
        landmarkNum = int(featureMaps[0].size()[1])
        # ~ landmarkNum = int(featureMaps[0].size()[1]/4)

        l, h, w = featureMaps[0].size()[2], featureMaps[0].size()[3], featureMaps[0].size()[4]
        # ~ print ("size: ", featureMaps[0].size())
        # ~ print ()
        lossOff = 0
        lossReg = 0
        loss = 0
        # print("1")
        X = np.round((landmarks[:, :, 0] * (h - 1)).numpy()).astype("int")
        Y = np.round((landmarks[:, :, 1] * (w - 1)).numpy()).astype("int")
        Z = np.round((landmarks[:, :, 2] * (l - 1)).numpy()).astype("int")

        for imageId in range(imageNum):
            for landmarkId in range(landmarkNum):
                # ~ print (Z[imageId][landmarkId], X[imageId][landmarkId], Y[imageId][landmarkId], self.HeatMap.size(), landmarkNum)
                # ~ print (l - Z[imageId][landmarkId], 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId], 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId], 2*w - Y[imageId][landmarkId])

                # ~ MyUtils.showDICOM(self.HeatMap[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]].detach().cpu().numpy(), X[imageId][landmarkId], Y[imageId][landmarkId], Z[imageId][landmarkId])

                self.binary_class_groundTruth[imageId, landmarkId, :, :, :] = self.HeatMap[
                                                                              l - Z[imageId][landmarkId]: 2 * l -
                                                                                                          Z[imageId][
                                                                                                              landmarkId],
                                                                              h - X[imageId][landmarkId]: 2 * h -
                                                                                                          X[imageId][
                                                                                                              landmarkId],
                                                                              w - Y[imageId][landmarkId]: 2 * w -
                                                                                                          Y[imageId][
                                                                                                              landmarkId]]
            # self.offsetMask = temMap + self.offsetMask

        indexs = self.binary_class_groundTruth > 0
        # indexs = self.offsetMask > 0
        # indexs = getMask(self.binary_class_groundTruth)
        # print("3")
        temloss = [
            [self.binaryLoss(featureMaps[0][imageId][landmarkId], self.binary_class_groundTruth[imageId][landmarkId])]
            # ~ , \

            # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*1][indexs[imageId][landmarkId]], \
            # ~ self.offsetMapx[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]]) , \

            # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*2][indexs[imageId][landmarkId]], \
            # ~ self.offsetMapy[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]]), \

            # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*3][indexs[imageId][landmarkId]], \
            # ~ self.offsetMapz[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]])]

            for imageId in range(imageNum)
            for landmarkId in range(landmarkNum)]
        loss1 = (sum([sum(temloss[ind]) for ind in range(imageNum * landmarkNum)])) / (imageNum * landmarkNum)
        # print("4")

        return loss1


class fusionLossFunc_improved_2(nn.Module):
    def __init__(self, use_gpu, R, imageSize, imageNum, landmarkNum):
        super(fusionLossFunc_improved_2, self).__init__()

        # ~ l, h, w = 96, 96, 96
        # ~ l, h, w = 48, 48, 48
        l, h, w = 64, 64, 64

        self.use_gpu = use_gpu
        self.R = R
        self.width = w
        self.higth = h
        self.Long = l

        self.binaryLoss = nn.BCEWithLogitsLoss(size_average=False)
        # ~ self.binaryLoss = nn.BCEWithLogitsLoss()

        self.huberLoss = torch.nn.L1Loss()
        # ~ self.offsetMask = torch.zeros(h, w).cuda(self.use_gpu)

        self.offsetMapx = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapy = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapz = np.ones((self.Long * 2, self.higth * 2, self.width * 2))

        self.HeatMap = np.zeros((self.Long * 2, self.higth * 2, self.width * 2))

        rr = R
        referPoint = (self.Long, self.higth, self.width)
        for k in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for i in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                for j in range(referPoint[2] - rr, referPoint[2] + rr + 1):
                    temdis = MyUtils.Mydist3D(referPoint, (k, i, j))
                    if temdis <= rr:
                        self.HeatMap[k][i][j] = 1

        for i in range(2 * self.Long):
            self.offsetMapz[i, :, :] = self.offsetMapz[i, :, :] * i

        for i in range(2 * self.higth):
            self.offsetMapx[:, i, :] = self.offsetMapx[:, i, :] * i

        for i in range(2 * self.width):
            self.offsetMapy[:, :, i] = self.offsetMapy[:, :, i] * i

        self.offsetMapz = referPoint[0] - self.offsetMapz
        self.offsetMapx = referPoint[1] - self.offsetMapx
        self.offsetMapy = referPoint[2] - self.offsetMapy

        # print (self.HeatMap)
        # print (self.offsetMapx)
        # print (self.offsetMapy)

        # ~ self.HeatMap = Variable(torch.from_numpy(self.HeatMap)).cuda(self.use_gpu).float()

        self.binary_class_groundTruth = torch.from_numpy(
            self.HeatMap[l - l // 2: 2 * l - l // 2, h - h // 2: 2 * h - h // 2, w - w // 2: 2 * w - w // 2]).cuda(
            self.use_gpu).float()

        # ~ self.offsetMapx = Variable(torch.from_numpy(self.offsetMapx)).cuda(self.use_gpu).float() / rr
        # ~ self.offsetMapy = Variable(torch.from_numpy(self.offsetMapy)).cuda(self.use_gpu).float() / rr
        # ~ self.offsetMapz = Variable(torch.from_numpy(self.offsetMapz)).cuda(self.use_gpu).float() / rr

        return

    def forward(self, featureMaps, landmarks):

        imageNum, landmarkNum, l, h, w = featureMaps.size()
        # ~ print ("size: ", featureMaps[0].size())
        # ~ print ()
        lossOff = 0
        lossReg = 0
        loss = 0

        indexs = self.binary_class_groundTruth > 0
        # print("3")
        temloss = [[self.binaryLoss(featureMaps[imageId][landmarkId], self.binary_class_groundTruth)]
                   # ~ , \

                   # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*1][indexs[imageId][landmarkId]], \
                   # ~ self.offsetMapx[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]]) , \

                   # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*2][indexs[imageId][landmarkId]], \
                   # ~ self.offsetMapy[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]]), \

                   # ~ self.huberLoss(featureMaps[0][imageId][landmarkId + landmarkNum*3][indexs[imageId][landmarkId]], \
                   # ~ self.offsetMapz[l - Z[imageId][landmarkId]: 2*l - Z[imageId][landmarkId], h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]][indexs[imageId][landmarkId]])]

                   for imageId in range(imageNum)
                   for landmarkId in range(landmarkNum)]
        loss2 = (sum([sum(temloss[ind]) for ind in range(imageNum * landmarkNum)])) / (imageNum * landmarkNum)
        # print("4")

        return loss2


class fusionLossFunc_improved_2_b(nn.Module):
    def __init__(self, use_gpu, R, imageSize, imageNum, landmarkNum):
        super(fusionLossFunc_improved_2_b, self).__init__()

        # ~ l, h, w = 96, 96, 96
        # ~ l, h, w = 48, 48, 48
        l, h, w = 64, 64, 64

        self.use_gpu = use_gpu
        self.R = R
        self.width = w
        self.higth = h
        self.Long = l

        self.binaryLoss = nn.BCEWithLogitsLoss(size_average=False)
        # ~ self.binaryLoss = nn.BCEWithLogitsLoss()

        self.huberLoss = torch.nn.L1Loss(size_average=False)
        # ~ self.offsetMask = torch.zeros(h, w).cuda(self.use_gpu)

        self.offsetMapx = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapy = np.ones((self.Long * 2, self.higth * 2, self.width * 2))
        self.offsetMapz = np.ones((self.Long * 2, self.higth * 2, self.width * 2))

        self.HeatMap = np.zeros((self.Long * 2, self.higth * 2, self.width * 2))

        rr = R
        referPoint = (self.Long, self.higth, self.width)
        for k in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for i in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                for j in range(referPoint[2] - rr, referPoint[2] + rr + 1):
                    temdis = MyUtils.Mydist3D(referPoint, (k, i, j))
                    if temdis <= rr:
                        self.HeatMap[k][i][j] = 1

        for i in range(2 * self.Long):
            self.offsetMapz[i, :, :] = self.offsetMapz[i, :, :] * i

        for i in range(2 * self.higth):
            self.offsetMapx[:, i, :] = self.offsetMapx[:, i, :] * i

        for i in range(2 * self.width):
            self.offsetMapy[:, :, i] = self.offsetMapy[:, :, i] * i

        self.offsetMapz = referPoint[0] - self.offsetMapz
        self.offsetMapx = referPoint[1] - self.offsetMapx
        self.offsetMapy = referPoint[2] - self.offsetMapy

        # print (self.HeatMap)
        # print (self.offsetMapx)
        # print (self.offsetMapy)

        # ~ self.HeatMap = Variable(torch.from_numpy(self.HeatMap)).cuda(self.use_gpu).float()

        self.binary_class_groundTruth = torch.from_numpy(
            self.HeatMap[l - l // 2: 2 * l - l // 2, h - h // 2: 2 * h - h // 2, w - w // 2: 2 * w - w // 2]).cuda(
            self.use_gpu).float()

        # ~ self.offsetMapx = torch.from_numpy(self.offsetMapx[l - l//2: 2*l - l//2, h - h//2: 2*h - h//2, w - w//2: 2*w - w//2]).cuda(self.use_gpu).float() / rr
        # ~ self.offsetMapy = torch.from_numpy(self.offsetMapy[l - l//2: 2*l - l//2, h - h//2: 2*h - h//2, w - w//2: 2*w - w//2]).cuda(self.use_gpu).float() / rr
        # ~ self.offsetMapz = torch.from_numpy(self.offsetMapz[l - l//2: 2*l - l//2, h - h//2: 2*h - h//2, w - w//2: 2*w - w//2]).cuda(self.use_gpu).float() / rr

        self.offsetMapx = torch.from_numpy(
            self.offsetMapx[l - l // 2: 2 * l - l // 2, h - h // 2: 2 * h - h // 2, w - w // 2: 2 * w - w // 2]).cuda(
            self.use_gpu).float() / 63
        self.offsetMapy = torch.from_numpy(
            self.offsetMapy[l - l // 2: 2 * l - l // 2, h - h // 2: 2 * h - h // 2, w - w // 2: 2 * w - w // 2]).cuda(
            self.use_gpu).float() / 63
        self.offsetMapz = torch.from_numpy(
            self.offsetMapz[l - l // 2: 2 * l - l // 2, h - h // 2: 2 * h - h // 2, w - w // 2: 2 * w - w // 2]).cuda(
            self.use_gpu).float() / 63

    def forward(self, featureMaps, landmarks):

        imageNum, landmarkNum, l, h, w = featureMaps.size()
        landmarkNum = int(landmarkNum / 4)
        # ~ print ("size: ", featureMaps[0].size())
        # ~ print ()
        lossOff = 0
        lossReg = 0
        loss = 0

        indexs = self.binary_class_groundTruth > 0
        # print("3")
        temloss = [[self.binaryLoss(featureMaps[imageId][landmarkId * 4], self.binary_class_groundTruth), \
 \
                    self.huberLoss(featureMaps[imageId][landmarkId * 4 + 1], \
                                   self.offsetMapx), \
 \
                    self.huberLoss(featureMaps[imageId][landmarkId * 4 + 2], \
                                   self.offsetMapy), \
 \
                    self.huberLoss(featureMaps[imageId][landmarkId * 4 + 3], \
                                   self.offsetMapz)]

                   for imageId in range(imageNum)
                   for landmarkId in range(landmarkNum)]
        loss2 = (sum([sum(temloss[ind]) for ind in range(imageNum * landmarkNum)])) / (imageNum * landmarkNum)
        # print("4")

        return loss2
