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

def getcropedInputs_related(ROIs, labels, inputs_origin, useGPU, index, config):
    # # 🔥 [DEBUG] 打印案发现场形状
    # if len(inputs_origin) > 0:
    #     print(f"[DEBUG 3 - CrashSite] inputs_origin[0] shape in MyUtils: {inputs_origin[0].shape}")
    
    labels_b = labels.detach().cpu().numpy()
    landmarks = ROIs
    landmarkNum = len(inputs_origin)

    b, c, l, h, w = inputs_origin[0].size()

    L, H, W = config.origin_image_size
    cropSize = 0
    if index == 0:
        cropSize = 32
    elif index == 1:
        cropSize = 16
    else:
        cropSize = 8

    # ~ print ("origin ", inputs_origin.size())

    X1, Y1, Z1 = landmarks[:, :, 0], landmarks[:, :, 1], landmarks[:, :, 2]
    X1, Y1, Z1 = np.round(X1 * (H - 1)).astype("int"), np.round(Y1 * (W - 1)).astype("int"), np.round(Z1 * (L - 1)).astype("int")

    X2, Y2, Z2 = labels_b[:, :, 0], labels_b[:, :, 1], labels_b[:, :, 2]
    X2, Y2, Z2 = np.round(X2 * (H - 1)).astype("int"), np.round(Y2 * (W - 1)).astype("int"), np.round(Z2 * (L - 1)).astype("int")

    X, Y, Z = X1 - X2 + int(h/2), Y1 - Y2 + int(w/2), Z1 - Z2 + int(l/2)
    # print(X, Y, Z)


    cropedDICOMs = []
    flag = True
    for landmarkId in range(landmarkNum):
        z, x, y = Z[0][landmarkId], X[0][landmarkId], Y[0][landmarkId]

        # if z<0 or z >= l or x < 0 or x >=h or y < 0 or y >= w:
        #     cropedDICOMs.append(torch.zeros(1, 1, 32, 32, 32))
        #     continue

        lz, uz, lx, ux, ly, uy = z - cropSize, z + cropSize, x - cropSize, x + cropSize, y - cropSize, y + cropSize
        lzz, uzz, lxx, uxx, lyy, uyy = max(lz, 0), min(uz, l), max(lx, 0), min(ux, h), max(ly, 0), min(uy, w)

        # ~ print (z, x, y)
        # ~ print ("boxes ", lz, uz, lx, ux, ly, uy)
        cropedDICOM = inputs_origin[landmarkId][:, :, lzz: uzz, lxx: uxx, lyy: uyy].clone()
        # ~ print ("check before", cropedDICOM.size())
        if lz < 0:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, 0 - lz, curentX, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((temTensor, cropedDICOM), 2)
        if uz > l:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, uz - l, curentX, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((cropedDICOM, temTensor), 2)
        if lx < 0:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, 0 - lx, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((temTensor, cropedDICOM), 3)
        if ux > h:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, ux - h, curentY)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((cropedDICOM, temTensor), 3)
        if ly < 0:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, curentX, 0 - ly)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((temTensor, cropedDICOM), 4)
        if uy > w:
            _, _, curentZ, curentX, curentY = cropedDICOM.size()
            temTensor = torch.zeros(b, c, curentZ, curentX, uy - w)
            if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
            cropedDICOM = torch.cat((cropedDICOM, temTensor), 4)

        # cropedDICOMs.append(cropedDICOM)
        cropedDICOMs.append(F.upsample(cropedDICOM, size=(32, 32, 32), mode='trilinear'))

    # ~ print (cropedDICOMs.size())
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
        # "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"  # 日志时间、脚本名、行数、消息
        "[%(levelname)s] %(message)s"           # 只保留级别+消息
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