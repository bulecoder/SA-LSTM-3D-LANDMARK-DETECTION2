from __future__ import print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import pandas as pd
import math
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom
import torch.nn.functional as F
import logging
import sys

def analysis_result(landmarkNum, Off):  # 可以处理缺失值（带NaN的情况）
    # 确保数据在 CPU 上以免占用显存，且方便计算
    if Off.is_cuda:
        Off = Off.cpu()

    thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    # 初始化输出矩阵
    SDR = torch.zeros((landmarkNum, len(thresholds)))
    SD = torch.zeros((landmarkNum))
    
    # 1. 计算 MRE (Mean Radial Error)
    MRE = torch.nanmean(Off, dim=0) 

    # 2. 计算 SDR 和 SD
    for landmarkId in range(landmarkNum):
        landmarkCol = Off[:, landmarkId]
        
        # 利用 torch.isnan 提取有效数据   ~torch.isnan() 表示取反，即“非NaN”
        valid_mask = ~torch.isnan(landmarkCol)
        valid_data = landmarkCol[valid_mask]
        
        if valid_data.numel() > 0: # 如果有有效数据
            # 计算标准差
            SD[landmarkId] = torch.std(valid_data)
            
            # 计算不同阈值下的成功率 SDR
            for i, th in enumerate(thresholds):
                SDR[landmarkId, i] = torch.le(valid_data, th).float().mean() # torch.le 是 <= (Less Equal)  .float().mean() 自动计算 True 的比例
        else:
            SD[landmarkId] = 0.0
            SDR[landmarkId, :] = 0.0

    return SDR, SD, MRE         # 返回的MRE是一个列表，代表7个关键点各自的平均误差


def analysis_result_overall(Off):
    """
    计算所有地标的整体统计指标
    Args:
        Off: 形状为 (N, landmarkNum) 的误差矩阵
    Returns:
        overall_SDR: 整体SDR (8个阈值)
        overall_SD: 整体标准差
        overall_MRE: 整体平均误差
    """
    # 将所有地标的误差展平
    all_errors = Off.flatten()
    
    # 计算整体MRE
    overall_MRE = np.mean(all_errors)
    
    # 计算整体SD
    overall_SD = np.sqrt(np.sum(np.power(all_errors - overall_MRE, 2)) / (len(all_errors) - 1))
    
    # 计算整体SDR
    overall_SDR = np.array([
        np.sum(all_errors <= 1) / len(all_errors),
        np.sum(all_errors <= 2) / len(all_errors),
        np.sum(all_errors <= 3) / len(all_errors),
        np.sum(all_errors <= 4) / len(all_errors),
        np.sum(all_errors <= 5) / len(all_errors),
        np.sum(all_errors <= 6) / len(all_errors),
        np.sum(all_errors <= 7) / len(all_errors),
        np.sum(all_errors <= 8) / len(all_errors)
    ])
    
    return overall_SDR, overall_SD, overall_MRE

def adjustment(ROIs, labels):
    temoff = (ROIs - labels)
    temoff[temoff > 0.055] = temoff[temoff > 0.055] * 0 + 0.055
    temoff[temoff < -0.055] = temoff[temoff < -0.055] * 0 - 0.055
    ROIs = labels + temoff
    return ROIs

def Mydist(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def Mydist3D(a, b):
    z1, x1, y1 = a
    z2, x2, y2 = b
    return math.sqrt((z2 - z1) ** 2 + (x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_coordinates_from_coarse_heatmaps(predicted_heatmap, global_coordinate):
    lent = len(predicted_heatmap)
    index = [2, 1, 0]       # 正确的索引
    global_coordinate_permute = global_coordinate.permute(3, 0, 1, 2)
    predict = [torch.sum((global_coordinate_permute * predicted_heatmap[i]).view(3, -1), dim = 1).unsqueeze(0) for i in range(lent)]
    predict = torch.cat(predict, dim=0)
    return predict[:, index]

def get_coordinates_from_fine_heatmaps(heatMaps, global_coordinate):
    lent = len(heatMaps)
    global_heatmap = [torch.sigmoid(heatMaps[i]) for i in range(lent)]
    global_heatmap = [global_heatmap[i] / global_heatmap[i].sum() for i in range(lent)]
    index = [1, 2, 0]
    global_coordinate_permute = global_coordinate.permute(3, 0, 1, 2)
    predict = [torch.sum((global_coordinate_permute * global_heatmap[i]).view(3, -1), dim = 1).unsqueeze(0) for i in range(lent)]
    predict = torch.cat(predict, dim=0)
    return predict[:, index]

def get_fine_errors(predicted_offset, labels, size_tensor):    # size_tensor 必须是物理尺寸 (mm)
    predict = predicted_offset * size_tensor.unsqueeze(1)
    labels_b = labels * size_tensor.unsqueeze(1)
    diff = predict - labels_b   # 计算差值(mm)
    tem_dist = torch.norm(diff, p=2, dim=2) # 计算欧氏距离
    return tem_dist # (B, N)

def get_coarse_errors(coarse_landmarks, labels, size_tensor):
    predict = coarse_landmarks * size_tensor.unsqueeze(1)
    labels_b = labels * size_tensor.unsqueeze(1)
    diff = predict - labels_b   # 计算差值(mm)
    tem_dist = torch.norm(diff, p=2, dim=2) # 计算欧氏距离
    return tem_dist

def get_global_feature(ROIs, coarse_feature, landmarkNum):
    # 原始代码：
    # X1, Y1, Z1 = ROIs[:, :, 0], ROIs[:, :, 1], ROIs[:, :, 2]
    # L, H, W = coarse_feature.size()[-3:]
    # X1, Y1, Z1 = np.round(X1 * (H - 1)).astype("int"), np.round(Y1 * (W - 1)).astype("int"), np.round(Z1 * (L - 1)).astype("int")
    # global_embedding = torch.cat([coarse_feature[:, :, Z1[0, i], X1[0, i], Y1[0, i]] for i in range(landmarkNum)], dim=0).unsqueeze(0)
    # return global_embedding

    # 原始代码预测结果可能会越界，这里进行优化
    # ROIs shape: [1, landmarkNum, 3]  coarse_feature shape: [B, C, L, H, W]
    # 1. 动态获取 feature map 的维度信息   L: Depth (Z), H: Height (X), W: Width (Y)
    L, H, W = coarse_feature.size()[-3:]
    
    # 2. 提取归一化坐标并进行维度安全限制 使用 np.clip 将坐标限制在 [0, 1] 之间，防止原始 ROIs 越界
    X1_norm = np.clip(ROIs[:, :, 0], 0, 1)
    Y1_norm = np.clip(ROIs[:, :, 1], 0, 1)
    Z1_norm = np.clip(ROIs[:, :, 2], 0, 1)
    
    # 3. 计算整数索引并再次确保不越界 (0 到 size-1)  np.round(val * (size - 1)) 能精准映射到最后一个像素中心
    X1 = np.round(X1_norm * (H - 1)).astype("int")
    Y1 = np.round(Y1_norm * (W - 1)).astype("int")
    Z1 = np.round(Z1_norm * (L - 1)).astype("int")
    
    # 4. 提取特征 使用列表推导式提取每个 landmark 对应的特征向量  coarse_feature[:, :, z, x, y] 提取的是 [B, C] 的特征
    global_embedding = torch.cat(
        [coarse_feature[:, :, Z1[0, i], X1[0, i], Y1[0, i]] for i in range(landmarkNum)], 
        dim=0
    ).unsqueeze(0)
    return global_embedding

# def getcropedInputs_related(ROIs, labels, inputs_origin, useGPU, index, config):
#     # # 🔥 [DEBUG] 打印案发现场形状
#     # if len(inputs_origin) > 0:
#     #     print(f"[DEBUG 3 - CrashSite] inputs_origin[0] shape in MyUtils: {inputs_origin[0].shape}")
    
#     labels_b = labels.detach().cpu().numpy()
#     landmarks = ROIs
#     landmarkNum = len(inputs_origin)

#     b, c, l, h, w = inputs_origin[0].size()

#     L, H, W = config.origin_image_size
#     cropSize = 0
#     if index == 0:
#         cropSize = 32
#     elif index == 1:
#         cropSize = 16
#     else:
#         cropSize = 8

#     # ~ print ("origin ", inputs_origin.size())

#     X1, Y1, Z1 = landmarks[:, :, 0], landmarks[:, :, 1], landmarks[:, :, 2]
#     X1, Y1, Z1 = np.round(X1 * (H - 1)).astype("int"), np.round(Y1 * (W - 1)).astype("int"), np.round(Z1 * (L - 1)).astype("int")

#     X2, Y2, Z2 = labels_b[:, :, 0], labels_b[:, :, 1], labels_b[:, :, 2]
#     X2, Y2, Z2 = np.round(X2 * (H - 1)).astype("int"), np.round(Y2 * (W - 1)).astype("int"), np.round(Z2 * (L - 1)).astype("int")

#     X, Y, Z = X1 - X2 + int(h/2), Y1 - Y2 + int(w/2), Z1 - Z2 + int(l/2)
#     # print(X, Y, Z)


#     cropedDICOMs = []
#     flag = True
#     for landmarkId in range(landmarkNum):
#         z, x, y = Z[0][landmarkId], X[0][landmarkId], Y[0][landmarkId]

#         # if z<0 or z >= l or x < 0 or x >=h or y < 0 or y >= w:
#         #     cropedDICOMs.append(torch.zeros(1, 1, 32, 32, 32))
#         #     continue

#         lz, uz, lx, ux, ly, uy = z - cropSize, z + cropSize, x - cropSize, x + cropSize, y - cropSize, y + cropSize
#         lzz, uzz, lxx, uxx, lyy, uyy = max(lz, 0), min(uz, l), max(lx, 0), min(ux, h), max(ly, 0), min(uy, w)

#         # ~ print (z, x, y)
#         # ~ print ("boxes ", lz, uz, lx, ux, ly, uy)
#         cropedDICOM = inputs_origin[landmarkId][:, :, lzz: uzz, lxx: uxx, lyy: uyy].clone()
#         # ~ print ("check before", cropedDICOM.size())
#         if lz < 0:
#             _, _, curentZ, curentX, curentY = cropedDICOM.size()
#             temTensor = torch.zeros(b, c, 0 - lz, curentX, curentY)
#             if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
#             cropedDICOM = torch.cat((temTensor, cropedDICOM), 2)
#         if uz > l:
#             _, _, curentZ, curentX, curentY = cropedDICOM.size()
#             temTensor = torch.zeros(b, c, uz - l, curentX, curentY)
#             if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
#             cropedDICOM = torch.cat((cropedDICOM, temTensor), 2)
#         if lx < 0:
#             _, _, curentZ, curentX, curentY = cropedDICOM.size()
#             temTensor = torch.zeros(b, c, curentZ, 0 - lx, curentY)
#             if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
#             cropedDICOM = torch.cat((temTensor, cropedDICOM), 3)
#         if ux > h:
#             _, _, curentZ, curentX, curentY = cropedDICOM.size()
#             temTensor = torch.zeros(b, c, curentZ, ux - h, curentY)
#             if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
#             cropedDICOM = torch.cat((cropedDICOM, temTensor), 3)
#         if ly < 0:
#             _, _, curentZ, curentX, curentY = cropedDICOM.size()
#             temTensor = torch.zeros(b, c, curentZ, curentX, 0 - ly)
#             if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
#             cropedDICOM = torch.cat((temTensor, cropedDICOM), 4)
#         if uy > w:
#             _, _, curentZ, curentX, curentY = cropedDICOM.size()
#             temTensor = torch.zeros(b, c, curentZ, curentX, uy - w)
#             if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
#             cropedDICOM = torch.cat((cropedDICOM, temTensor), 4)

#         # cropedDICOMs.append(cropedDICOM)
#         cropedDICOMs.append(F.upsample(cropedDICOM, size=(32, 32, 32), mode='trilinear'))

#     # ~ print (cropedDICOMs.size())
#     return cropedDICOMs


# MyUtils.py 中的 getcropedInputs_related 函数

def getcropedInputs_related(ROIs, labels, inputs_origin, useGPU, index, config):
    """
    针对 Full Image 的极简切图函数
    直接根据 ROIs 在原图上切出 patch
    """
    # 1. 准备图像数据 (Tensor)
    img_tensor = inputs_origin[0]

    # 维度兼容性处理: (C, D, H, W) -> (1, C, D, H, W)
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(0)

    # 现在的 img_tensor 保证是 5 维 (B, C, D, H, W)
    b, c, D, H, W = img_tensor.size()
    
    # 2. 确定 Crop Size
    base_size = 64 # 根据 config.crop_size 调整
    if index == 0:   crop_r = base_size // 2      # r=32
    elif index == 1: crop_r = base_size // 4      # r=16
    else:            crop_r = base_size // 8      # r=8

    # 3. 计算中心坐标 (反归一化)
    # ROIs 可能传进来是 (Batch, N, 3)，这里取 Batch 0
    current_rois = ROIs[0] # (N, 3)
    landmarkNum = current_rois.shape[0]

    L_o, H_o, W_o = config.origin_image_size
    
    # 🔥 [修复核心] 兼容 Tensor 和 Numpy 输入
    if isinstance(current_rois, torch.Tensor):
        x_raw = current_rois[:, 0].detach().cpu().numpy()
        y_raw = current_rois[:, 1].detach().cpu().numpy()
        z_raw = current_rois[:, 2].detach().cpu().numpy()
    else:
        x_raw = current_rois[:, 0]
        y_raw = current_rois[:, 1]
        z_raw = current_rois[:, 2]
    
    # 计算像素坐标 (假设 ROIs 对应 W, H, D 即 X, Y, Z)
    # 注意：请确保你的 ROIs 坐标定义和图像维度是一致的
    cX = np.round(x_raw * (W_o - 1)).astype(int)
    cY = np.round(y_raw * (H_o - 1)).astype(int)
    cZ = np.round(z_raw * (L_o - 1)).astype(int)

    cropedDICOMs = []
    
    # 4. 开始切图
    for i in range(landmarkNum):
        # 提取中心点 (PyTorch Tensor 顺序通常是 D, H, W -> z, y, x)
        z, y, x = cZ[i], cY[i], cX[i]
        
        # 计算边界
        lz, uz = z - crop_r, z + crop_r
        ly, uy = y - crop_r, y + crop_r
        lx, ux = x - crop_r, x + crop_r
        
        # 钳位边界 (用于 Slice)
        lzz, uzz = max(lz, 0), min(uz, D)
        lyy, uyy = max(ly, 0), min(uy, H)
        lxx, uxx = max(lx, 0), min(ux, W)
        
        # 切片 (这里需要 5 维数据)
        patch = img_tensor[:, :, lzz:uzz, lyy:uyy, lxx:uxx].clone()
        
        # Padding (如果切出界了补零)
        pad_z_l = abs(lz) if lz < 0 else 0
        pad_z_r = (uz - D) if uz > D else 0
        pad_y_l = abs(ly) if ly < 0 else 0
        pad_y_r = (uy - H) if uy > H else 0
        pad_x_l = abs(lx) if lx < 0 else 0
        pad_x_r = (ux - W) if ux > W else 0
        
        if (pad_x_l+pad_x_r+pad_y_l+pad_y_r+pad_z_l+pad_z_r) > 0:
            # F.pad顺序: x_l, x_r, y_l, y_r, z_l, z_r
            patch = torch.nn.functional.pad(patch, (pad_x_l, pad_x_r, pad_y_l, pad_y_r, pad_z_l, pad_z_r))

        # 统一 Resize (确保输出尺寸一致)
        target_size = (64, 64, 64) 
        if patch.shape[2:] != target_size:
            patch = torch.nn.functional.interpolate(patch, size=target_size, mode='trilinear', align_corners=False)
            
        cropedDICOMs.append(patch)

    return cropedDICOMs

def getcropedInputs(ROIs, inputs_origin, cropSize, useGPU):
    # ROIs: (1, N, 3) 绝对像素坐标 (已在 MyDataLoader 中钳位)
    # inputs_origin: (B, C, D, H, W)
    
    landmarks = ROIs
    landmarkNum = landmarks.shape[1]
    b, c, l, h, w = inputs_origin.size()

    # cropSize 传入的是直径 (96)，计算半径
    radius = int(cropSize / 2)
    
    # 🔥 关键修改：直接使用像素坐标，移除 * (h-1) 的缩放
    X = landmarks[:, :, 0]
    Y = landmarks[:, :, 1]
    Z = landmarks[:, :, 2]
    
    # 转整型
    X = np.round(X).astype("int")
    Y = np.round(Y).astype("int")
    Z = np.round(Z).astype("int")
    
    cropedDICOMs = []
    
    for landmarkId in range(landmarkNum):
        # 注意：这里假设输入的 ROIs 顺序是 (X, Y, Z) 对应 (H, W, D) 还是 (D, H, W)?
        # 根据之前的报错 "allocate ... uy - w"，以及 MyDataLoader 里的 reshape
        # 我们假设输入顺序已经适配了
        
        # MyDataLoader 传入的是 (D, H, W) 对应的坐标
        # 原代码看起来 X 对应 h, Y 对应 w, Z 对应 l
        z, x, y = Z[0][landmarkId], X[0][landmarkId], Y[0][landmarkId]
        
        lz, uz = z - radius, z + radius
        lx, ux = x - radius, x + radius
        ly, uy = y - radius, y + radius
        
        # 计算有效区域 (Clamp)
        lzz, uzz = max(lz, 0), min(uz, l)
        lxx, uxx = max(lx, 0), min(ux, h)
        lyy, uyy = max(ly, 0), min(uy, w)

        # 切取有效部分
        cropedDICOM = inputs_origin[:, :, lzz: uzz, lxx: uxx, lyy: uyy].clone()
        
        # Padding 逻辑 (处理边缘)
        # 如果 MyDataLoader 已经做了 Safe Clamp，这里其实不会触发 Padding
        # 但保留以防万一
        
        # Z轴 padding
        if lz < 0:
            pad = torch.zeros(b, c, 0 - lz, cropedDICOM.size(3), cropedDICOM.size(4)).to(inputs_origin.device)
            cropedDICOM = torch.cat((pad, cropedDICOM), 2)
        if uz > l:
            pad = torch.zeros(b, c, uz - l, cropedDICOM.size(3), cropedDICOM.size(4)).to(inputs_origin.device)
            cropedDICOM = torch.cat((cropedDICOM, pad), 2)
            
        # X轴 padding
        if lx < 0:
            pad = torch.zeros(b, c, cropedDICOM.size(2), 0 - lx, cropedDICOM.size(4)).to(inputs_origin.device)
            cropedDICOM = torch.cat((pad, cropedDICOM), 3)
        if ux > h:
            pad = torch.zeros(b, c, cropedDICOM.size(2), ux - h, cropedDICOM.size(4)).to(inputs_origin.device)
            cropedDICOM = torch.cat((cropedDICOM, pad), 3)
            
        # Y轴 padding
        if ly < 0:
            pad = torch.zeros(b, c, cropedDICOM.size(2), cropedDICOM.size(3), 0 - ly).to(inputs_origin.device)
            cropedDICOM = torch.cat((pad, cropedDICOM), 4)
        if uy > w:
            pad = torch.zeros(b, c, cropedDICOM.size(2), cropedDICOM.size(3), uy - w).to(inputs_origin.device)
            cropedDICOM = torch.cat((cropedDICOM, pad), 4)

        cropedDICOMs.append(cropedDICOM)

    return cropedDICOMs

def get_local_patches(ROIs, cropedtems, base_coordinate, usegpu):
    local_coordinate = []
    local_patches = []
    for i in range(len(cropedtems)):
        centre = torch.from_numpy(ROIs[0, i, :]).cuda(usegpu)
        tem = base_coordinate + centre
        local_coordinate.append(tem)
    local_patches = cropedtems
    return local_patches, local_coordinate

def getCroped(ROIs, outputs):
    # imageNum, landmarkNum * channels, Long, Height, Width
    Y1, Y2, Y3 = outputs[1], outputs[2], outputs[3]
    size1, size2, size3 = Y1.size()[2:], Y2.size()[2:], Y3.size()[2:]
    print(size1, size2, size3)

def resizeDICOM(DICOM, shape_DICOM):
    l, h, w = DICOM.shape[:3]
    newl, newh, neww = shape_DICOM
    scalel, scaleh, scalew = newl / l, newh / h, neww / w

    newDICOM = zoom(DICOM, (scalel, scaleh, scalew))
    print(newDICOM.shape)
    return newDICOM

def showDICOM(DICOM, label, predict, epoch, lent):

    # import pdb
    # pdb.set_trace()

    x, y, z = int(label[0] * 767), int(label[1] * 767), int(label[2] * 575)
    xx, yy, zz = int(predict[0] * 767), int(predict[1] * 767), int(predict[2] * 575)

    imageX = DICOM[:, x, :]
    imageY = DICOM[:, :, y]
    imageZ = DICOM[z, :, :]
    # ~ print (x, y, z)
    # ~ print ("imageX", imageX.shape)
    # ~ print ("imageY", imageY.shape)
    # ~ print ("imageZ", imageZ.shape)

    minvX, maxvX = np.min(imageX), np.max(imageX)
    minvY, maxvY = np.min(imageY), np.max(imageY)
    minvZ, maxvZ = np.min(imageZ), np.max(imageZ)

    imageX = (imageX - minvX) / (maxvX - minvX) * 255
    imageY = (imageY - minvY) / (maxvY - minvY) * 255
    imageZ = (imageZ - minvZ) / (maxvZ - minvZ) * 255

    imageX = Image.fromarray(imageX.astype('uint8'))
    imageX = imageX.convert('RGB')
    drawX = ImageDraw.Draw(imageX)

    imageY = Image.fromarray(imageY.astype('uint8'))
    imageY = imageY.convert('RGB')
    drawY = ImageDraw.Draw(imageY)

    imageZ = Image.fromarray(imageZ.astype('uint8'))



    imageZ = imageZ.convert('RGB')
    drawZ = ImageDraw.Draw(imageZ)
    r = int(DICOM.shape[0] / 80)


    positionX = (y - r, z - r, y + r, z + r)
    positionY = (x - r, z - r, x + r, z + r)
    positionZ = (y - r, x - r, y + r, x + r)

    positionXX = (yy - r, zz - r, yy + r, zz + r)
    positionYY = (xx - r, zz - r, xx + r, zz + r)
    positionZZ = (yy - r, xx - r, yy + r, xx + r)

    drawX.ellipse(positionXX, fill=(255, 0, 0))
    drawY.ellipse(positionYY, fill=(255, 0, 0))
    drawZ.ellipse(positionZZ, fill=(255, 0, 0))

    drawX.ellipse(positionX, fill=(0, 255, 0))
    drawY.ellipse(positionY, fill=(0, 255, 0))
    drawZ.ellipse(positionZ, fill=(0, 255, 0))


    imageX.save("vis_images/" + str(lent) + "_" + str(epoch) + "_imageX.jpg")
    imageY.save("vis_images/" + str(lent) + "_" + str(epoch) + "_imageY.jpg")
    imageZ.save("vis_images/" + str(lent) + "_" + str(epoch) + "_imageZ.jpg")

    # plt.suptitle("multi_image")
    # plt.subplot(1, 3, 1), plt.title("x")
    # plt.imshow(imageX, cmap='gray', interpolation='nearest'), plt.axis("off")
    # plt.subplot(1, 3, 2), plt.title("y")
    # plt.imshow(imageY, cmap='gray', interpolation='nearest'), plt.axis("off")
    # plt.subplot(1, 3, 3), plt.title("z")
    # plt.imshow(imageZ, cmap='gray', interpolation='nearest')
    # plt.savefig("filename.png")
    # print("filename.png")
    # plt.show()
    # dfdf = input()

def drawImage(image, coordindates_before, coordindates_after):
    # image = image_before
    # image = Image.fromarray((image * 255).astype('uint8'))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 15)
    t = 0

    for ide in range(68):
        r = 6
        t = t + 1
        # draw.rectangle(coordindates_before, outline = "red")
        # x, y = coordindates_after[ide]['x'], coordindates_after[ide]['y']
        x, y = coordindates_after[ide][0], coordindates_after[ide][1]
        position = (x - r, y - r, x + r, y + r)

        # draw.ellipse(position,fill = (0, 255, 0))

        draw.text((x, y), str(t), fill=(0, 255, 255), font=font)

    plt.imshow(image, cmap='gray', interpolation='nearest')
    image.save("compare.png")
    fdf = input()


# return image
def Mydist(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def Mydist3D(a, b):
    z1, x1, y1 = a
    z2, x2, y2 = b
    return math.sqrt((z2 - z1) ** 2 + (x2 - x1) ** 2 + (y2 - y1) ** 2)

def getcropedInputs(ROIs, inputs_origin, cropSize, useGPU):
    # ROIs: (1, N, 3) 绝对像素坐标 (已在 MyDataLoader 中钳位)
    # inputs_origin: (B, C, D, H, W)
    
    landmarks = ROIs
    landmarkNum = landmarks.shape[1]
    b, c, l, h, w = inputs_origin.size()

    # cropSize 传入的是直径 (96)，计算半径
    radius = int(cropSize / 2)
    
    # 🔥 关键修改：直接使用像素坐标，移除 * (h-1) 的缩放
    X = landmarks[:, :, 0]
    Y = landmarks[:, :, 1]
    Z = landmarks[:, :, 2]
    
    # 转整型
    X = np.round(X).astype("int")
    Y = np.round(Y).astype("int")
    Z = np.round(Z).astype("int")
    
    cropedDICOMs = []
    
    for landmarkId in range(landmarkNum):
        # 注意：这里假设输入的 ROIs 顺序是 (X, Y, Z) 对应 (H, W, D) 还是 (D, H, W)?
        # 根据之前的报错 "allocate ... uy - w"，以及 MyDataLoader 里的 reshape
        # 我们假设输入顺序已经适配了
        
        # MyDataLoader 传入的是 (D, H, W) 对应的坐标
        # 原代码看起来 X 对应 h, Y 对应 w, Z 对应 l
        z, x, y = Z[0][landmarkId], X[0][landmarkId], Y[0][landmarkId]
        
        lz, uz = z - radius, z + radius
        lx, ux = x - radius, x + radius
        ly, uy = y - radius, y + radius
        
        # 计算有效区域 (Clamp)
        lzz, uzz = max(lz, 0), min(uz, l)
        lxx, uxx = max(lx, 0), min(ux, h)
        lyy, uyy = max(ly, 0), min(uy, w)

        # 切取有效部分
        cropedDICOM = inputs_origin[:, :, lzz: uzz, lxx: uxx, lyy: uyy].clone()
        
        # Padding 逻辑 (处理边缘)
        # 如果 MyDataLoader 已经做了 Safe Clamp，这里其实不会触发 Padding
        # 但保留以防万一
        
        # Z轴 padding
        if lz < 0:
            pad = torch.zeros(b, c, 0 - lz, cropedDICOM.size(3), cropedDICOM.size(4)).to(inputs_origin.device)
            cropedDICOM = torch.cat((pad, cropedDICOM), 2)
        if uz > l:
            pad = torch.zeros(b, c, uz - l, cropedDICOM.size(3), cropedDICOM.size(4)).to(inputs_origin.device)
            cropedDICOM = torch.cat((cropedDICOM, pad), 2)
            
        # X轴 padding
        if lx < 0:
            pad = torch.zeros(b, c, cropedDICOM.size(2), 0 - lx, cropedDICOM.size(4)).to(inputs_origin.device)
            cropedDICOM = torch.cat((pad, cropedDICOM), 3)
        if ux > h:
            pad = torch.zeros(b, c, cropedDICOM.size(2), ux - h, cropedDICOM.size(4)).to(inputs_origin.device)
            cropedDICOM = torch.cat((cropedDICOM, pad), 3)
            
        # Y轴 padding
        if ly < 0:
            pad = torch.zeros(b, c, cropedDICOM.size(2), cropedDICOM.size(3), 0 - ly).to(inputs_origin.device)
            cropedDICOM = torch.cat((pad, cropedDICOM), 4)
        if uy > w:
            pad = torch.zeros(b, c, cropedDICOM.size(2), cropedDICOM.size(3), uy - w).to(inputs_origin.device)
            cropedDICOM = torch.cat((cropedDICOM, pad), 4)

        cropedDICOMs.append(cropedDICOM)

    return cropedDICOMs

# 使用log来记录，同时建立两条通道，一条通往.log文件，一条通往屏幕
def get_logger(filename, verbosity=1, name=None):
    """
    创建一个日志记录器 logger
    :param filename: 日志文件保存路径 (例如: runs/exp1/train.log)
    :param verbosity: 日志级别
    :param name: logger 的名字
    :return: 配置好的 logger 对象
    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    # 1. File Handler (写入文件)
    fh = logging.FileHandler(filename, "w", encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 2. Stream Handler (输出到终端/屏幕)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

class GPUAugmentor:
    """
    在 GPU 上对 3D 图像进行实时增强 (旋转、缩放、强度变换)
    """
    def __init__(self, device, angle_range=(-10, 10), scale_range=(0.9, 1.1)):
        self.device = device
        self.angle_range = angle_range
        self.scale_range = scale_range

    def __call__(self, images, landmarks):
        """
        :param images: (B, 1, D, H, W) Tensor
        :param landmarks: (B, N, 3) Tensor, 坐标顺序必须是 (z, y, x) 对应 (D, H, W)
        """
        current_device = images.device
        B, C, D, H, W = images.shape
        
        # --- 1. 随机参数生成 ---
        # TODO:旋转角度 (弧度) - 目前只做 Z 轴旋转 (平面内旋转)，这是最关键的
        angles = (torch.rand(B, device=current_device) * (self.angle_range[1] - self.angle_range[0]) + self.angle_range[0])
        rads = torch.deg2rad(-angles) # 取反适配 grid_sample 方向

        # 缩放因子
        scales = (torch.rand(B, device=current_device) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0])

        # --- 2. 构建仿射变换矩阵 (B, 3, 4) ---
        # 目标是构建一个矩阵，将像素网格进行旋转和缩放
        theta = torch.zeros(B, 3, 4, device=current_device)
        
        cos_a = torch.cos(rads)
        sin_a = torch.sin(rads)

        # 缩放 + 旋转 (绕 D 轴 / Z 轴)
        # 矩阵结构:
        # [ sc,  0,   0,   0 ]
        # [ 0,   c*s, -s*s, 0 ]
        # [ 0,   s*s, c*s, 0 ]
        
        # D 轴 (Depth) 只缩放，不旋转
        theta[:, 0, 0] = scales 
        
        # H, W 平面 (Height, Width) 进行旋转 + 缩放
        theta[:, 1, 1] = scales * cos_a
        theta[:, 1, 2] = -scales * sin_a * (W / H) # 修正宽高比，防止旋转后变形
        theta[:, 2, 1] = scales * sin_a * (H / W)
        theta[:, 2, 2] = scales * cos_a

        # --- 3. 应用几何变换 (Grid Sample) ---
        grid = F.affine_grid(theta, images.size(), align_corners=False)
        aug_images = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # --- 4. 应用关键点变换 (矩阵乘法) ---
        # 关键点旋转中心 (图像中心)
        center = torch.tensor([D/2, H/2, W/2], device=current_device)
        
        # 构造对应的旋转矩阵 R (B, 3, 3)
        R = torch.zeros(B, 3, 3, device=current_device)
        R[:, 0, 0] = 1 
        R[:, 1, 1] = cos_a
        R[:, 1, 2] = -sin_a
        R[:, 2, 1] = sin_a
        R[:, 2, 2] = cos_a
        
        # 坐标变换公式: (P - Center) @ R.T * Scale + Center
        # 注意：这里我们是在物理坐标系下操作，不需要像 grid_sample 那样考虑宽高比修正
        landmarks = (landmarks - center)
        landmarks = torch.bmm(landmarks, R.transpose(1, 2)) 
        landmarks = landmarks * scales.unsqueeze(1).unsqueeze(2) + center

        # --- 5. 强度/对比度变换 (Intensity Shift) ---
        if torch.rand(1) < 0.5:
            contrast = torch.rand(B, 1, 1, 1, 1, device=current_device) * 0.4 + 0.8 # 0.8 ~ 1.2
            brightness = torch.rand(B, 1, 1, 1, 1, device=current_device) * 0.2 - 0.1 # -0.1 ~ 0.1
            aug_images = aug_images * contrast + brightness
            aug_images = torch.clamp(aug_images, 0.0, 1.0) # 保持归一化

        return aug_images, landmarks
    
# 一键处理函数 (Clean Wrapper)
def prepare_batch_input(data, config, phase, augmentor=None):
    """
    输入原始 Batch 数据，输出模型可直接用的 Coarse输入 和 Fine输入
    """
    # 1. 搬运到 GPU
    inputs_origin = data['DICOM_origin'].cuda(config.use_gpu) # (B, D, H, W)
    if len(inputs_origin.shape) == 3: inputs_origin = inputs_origin.unsqueeze(0).unsqueeze(0)
    elif len(inputs_origin.shape) == 4: inputs_origin = inputs_origin.unsqueeze(1) # (B, 1, D, H, W)
    
    labels = data['landmarks'].cuda(config.use_gpu).float()

    # 2. 训练阶段执行增强
    if phase == 'train' and augmentor is not None:
        inputs_origin, labels = augmentor(inputs_origin, labels)
        
        # 安全钳位
        D, H, W = inputs_origin.shape[2:]
        labels[:, :, 0] = torch.clamp(labels[:, :, 0], 0, D-1)
        labels[:, :, 1] = torch.clamp(labels[:, :, 1], 0, H-1)
        labels[:, :, 2] = torch.clamp(labels[:, :, 2], 0, W-1)

    # 3. GPU 生成 Coarse 输入 (下采样)
    inputs_coarse = torch.nn.functional.interpolate(inputs_origin, size=config.image_scale, mode='trilinear', align_corners=False)

    # 4. 格式适配 (Hack)
    # 因为你的 fine_LSTM 内部还在用 CPU 切图，我们需要把增强后的高清图转回 CPU list
    # 虽然多了一步传输，但依然比 CPU 旋转快得多
    inputs_origin_list = [inputs_origin[i].detach().cpu() for i in range(inputs_origin.shape[0])]

    _, _, D, H, W = inputs_origin.shape

    size_tensor = torch.tensor([D, H, W], device=labels.device).float()

    labels = labels / size_tensor

    return inputs_coarse, inputs_origin_list, labels