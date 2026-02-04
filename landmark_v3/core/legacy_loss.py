from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import math
from utils import legacy_utils as MyUtils
import torch.nn.functional as F

class coarse_heatmap(nn.Module):
    def __init__(self, config):
        """
        初始化：保持与原代码一致的参数接口
        """
        super(coarse_heatmap, self).__init__()
        self.use_gpu = config.use_gpu
        self.landmarkNum = config.landmarkNum
        self.depth, self.height, self.width = config.image_scale
        self.sigma = getattr(config, 'sigma', 2.0)  # sigma，默认用 2 (与你旧代码的 dev=2 一致)

    def generate_target_heatmap(self, labels_pixel, batch_size, device):
        """
        [内部函数] 现场生成高斯热图，替代旧代码的 self.HeatMap_groundTruth 裁剪逻辑
        修复了旧代码中 X/Y 轴映射错误的 Bug。
        """
        # 1. 创建坐标网格 (D, H, W)
        # indexing='ij' 意味着维度顺序是 (Axis 0, Axis 1, Axis 2) -> (D, H, W)
        z = torch.arange(self.depth, device=device).float()
        h = torch.arange(self.height, device=device).float()
        w = torch.arange(self.width, device=device).float()
        
        grid_z, grid_h, grid_w = torch.meshgrid(z, h, w, indexing='ij')
        
        targets = []
        
        for b in range(batch_size):
            batch_targets = []
            for n in range(self.landmarkNum):
                # labels_pixel 是 [x, y, z] -> [Width, Height, Depth]
                real_x = labels_pixel[b, n, 0] # Width
                real_y = labels_pixel[b, n, 1] # Height
                real_z = labels_pixel[b, n, 2] # Depth
                
                # [Mask Check] 缺失值处理：生成全黑图 (与旧代码逻辑一致)
                if labels_pixel[b, n, 0] < 0: 
                    batch_targets.append(torch.zeros_like(grid_z))
                    continue
                
                # [生成高斯]
                # 关键修复：正确对应坐标轴
                # grid_w 对应 Width (x), grid_h 对应 Height (y), grid_z 对应 Depth (z)
                dist_sq = (grid_w - real_x)**2 + (grid_h - real_y)**2 + (grid_z - real_z)**2
                
                # 生成未归一化的高斯
                heatmap = torch.exp(-dist_sq / (2 * self.sigma**2))
                
                # [归一化] 
                # 旧代码中: coarse_heatmap_gt / sum
                # CoarseNet 输出的是概率分布（Sum=1），所以 GT 也必须 Sum=1
                heatmap_sum = heatmap.sum()
                if heatmap_sum > 0:
                    heatmap = heatmap / heatmap_sum
                
                batch_targets.append(heatmap)
            
            # Stack landmarks: (N, D, H, W)
            targets.append(torch.stack(batch_targets))
            
        # Stack batch: (B, N, D, H, W)
        return torch.stack(targets)

    def forward(self, predicted_heatmap, local_coordinate, labels, phase):
        """
        前向传播：保持接口完全不变
        predicted_heatmap: list of tensors [(B, D, H, W), ...]
        local_coordinate: 暂时没用到 (保持接口兼容)
        labels: (B, N, 3) 归一化坐标
        phase: 暂时没用到
        """
        batch_size = labels.shape[0]
        
        # 1. [适配旧输入] 将 list 转为 tensor
        # CoarseNet 输出是 list，这里 stack 起来方便并行计算
        # Shape: (B, N, D, H, W)
        pred_tensor = torch.stack(predicted_heatmap, dim=1)
        
        # 2. [坐标转换] 归一化 -> 像素坐标
        # labels 是 [x, y, z]，对应 [Width, Height, Depth]
        scale = torch.tensor([self.width-1, self.height-1, self.depth-1], device=labels.device)
        labels_pixel = labels * scale

        # 3. [生成真值]
        with torch.no_grad(): # GT 生成不需要梯度
            target_heatmap = self.generate_target_heatmap(labels_pixel, batch_size, labels.device)
        
        # 4. [计算 Loss]
        # Mask: 找出有效点 (x >= 0)
        # shape: (B, N, 1, 1, 1) 以便广播
        mask = (labels[:, :, 0] >= 0).view(batch_size, self.landmarkNum, 1, 1, 1).float()
        
        # [核心改动] 使用 MSE Loss (L2) 替代 L1 Loss
        # 原因：对于高斯热图回归，MSE 通常比 L1 收敛更稳、更快，且对峰值更敏感。
        # 同时保留了 Mask 机制
        # loss = (pred_tensor - target_heatmap) ** 2 
        loss = abs(pred_tensor - target_heatmap)    # 暂时先使用L1 Loss
        
        # 只对有效点求和
        total_loss = (loss * mask).sum()
        
        # 归一化 Loss：除以有效点的数量 (防止 batch size 变化导致 loss 波动)
        # 注意：这里分母加了 epsilon 防止除零
        valid_points_count = mask.sum() + 1e-8
        
        return total_loss / valid_points_count

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