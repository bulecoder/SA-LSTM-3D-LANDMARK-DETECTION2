import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# --- 🔥 修改 import 路径 ---
# 假设原来的 MyNetworkLayer.py 内容也整合进来了，或者是单独的文件
# 如果你打算合并文件，把 MNL 的内容贴在这里。
# 如果不想合并，请把 MyNetworkLayer.py 放到 models/ 目录下并改名为 layers.py
# 这里假设你保留了 MyNetworkLayer 的独立性 (推荐)
from . import legacy_layers as MNL  # 假设你把 MyNetworkLayer 重命名为 legacy_layers.py 放在同级目录
from utils import legacy_utils as MyUtils # 引用旧的工具箱

class coarseNet(nn.Module):
    def __init__(self, config):
        super(coarseNet, self).__init__()
        self.landmarkNum = config.landmarkNum
        self.usegpu = config.use_gpu
        self.image_scale = config.image_scale
        self.u_net = MNL.U_Net3D(1, 64)
        self.dropout = nn.Dropout3d(p=0.5)  # 30%概率丢弃特征，正则化抑制过拟合
        self.conv3d = nn.Sequential(
            nn.Conv3d(64, config.landmarkNum, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 1. 骨干网络提取特征
        global_features = self.u_net(x)  # x: (B, 1, D, H, W)
        global_features_drop = self.dropout(global_features)    # 在进入最后的预测层之前，应用dropout
        # 2. Conv3D+激活+防除零（优化epsilon，统一为1e-8）
        x = self.conv3d(global_features_drop)  # x: (B, landmarkNum, D, H, W)
        epsilon = 1e-9
        x = x + epsilon  # 替换原代码的+1e-9，统一epsilon
        # 3. 修复维度展平+求和（保留批次维度，核心修复）
        batch_size = x.shape[0]  # 动态获取批次大小，适配任意B
        flat_x = x.view(batch_size, self.landmarkNum, -1)  # (B, landmarkNum, D*H*W)
        heatmap_sum = torch.sum(flat_x, dim=2)  # 仅对空间维度求和，shape: (B, landmarkNum)
        # 5. 归一化（保留列表推导式，修复广播除法）
        global_heatmap = [
            # x[:,i,...]：适配任意批次，替换原x[0,i,...]
            # heatmap_sum[:,i].view(...)：扩充维度实现广播除法，适配(B,D,H,W)
            x[:, i, :, :, :] / heatmap_sum[:, i].view(batch_size, 1, 1, 1)
            for i in range(self.landmarkNum)
        ]
        return global_heatmap, global_features      # 返回的 features 是原始的 global_features，没有加dropout，这样传给fineNet的特征是完整的，不会缺信息

class fine_LSTM(nn.Module):
    def __init__(self, config):
        super(fine_LSTM, self).__init__()

        # landmarkNum, use_gpu, iteration, cropSize

        self.landmarkNum = config.landmarkNum
        self.usegpu = config.use_gpu
        self.encoder = MNL.U_Net3D_encoder(1, 64)
        self.iteration = config.iteration
        self.crop_size = config.crop_size
        self.origin_image_size = config.origin_image_size
        self.config = config

        w, h, l = self.origin_image_size
        # (576, 768, 768)

        self.size_tensor = torch.tensor([1 / (l - 1), 1 / (h - 1), 1 / (w - 1)]).cuda(self.usegpu)
        # 这里的输入维度是 512 (Local) + 64 (Global) = 576
        self.decoders_offset_x = nn.Conv1d(self.landmarkNum, self.landmarkNum, 512 + 64, 1, 0, groups=self.landmarkNum)
        self.decoders_offset_y = nn.Conv1d(self.landmarkNum, self.landmarkNum, 512 + 64, 1, 0, groups=self.landmarkNum)
        self.decoders_offset_z = nn.Conv1d(self.landmarkNum, self.landmarkNum, 512 + 64, 1, 0, groups=self.landmarkNum)


        self.attention_gate_share = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.Tanh(),
            # nn.Linear(256, 1)
            # nn.Conv1d(landmarkNum, landmarkNum, 256, 1, 0, groups=landmarkNum),
        )
        self.attention_gate_head = nn.Conv1d(self.landmarkNum, self.landmarkNum, 256, 1, 0, groups=self.landmarkNum)
        self.graph_attention = MNL.graph_attention(64, self.usegpu)
        # self.graph_attention = MNL.graph_attention(512 + 64, self.usegpu)   # 将全局特征和局部特征进行拼接后，一起送入GNN进行交互

    def forward(self, coarse_landmarks, labels, inputs_origin, coarse_feature, phase, size_tensor_inv):

        # cropedtems = MyUtils.getcropedInputs_related(ROIs, labels, inputs_origin, -1, 0)
        # cropedtems = torch.cat([cropedtems[i].cuda(self.usegpu) for i in range(len(cropedtems))], dim=0)
        # features = self.encoder(cropedtems).squeeze().unsqueeze(0)
        # global_feature = MyUtils.get_global_feature(ROIs, coarse_feature)
        # global_feature = self.graph_attention(ROIs, global_feature)
        # features = torch.cat((features, global_feature),dim=2)
        # x, y, z = self.decoders_offset_x(features), self.decoders_offset_y(features), self.decoders_offset_z(features)
        # predict = torch.cat([x, y, z], dim=2) * self.size_tensor.cuda(self.usegpu) + torch.from_numpy(ROIs).cuda(self.usegpu)

        h_state = 0
        predicts = []
        c_state = 0
        predict = coarse_landmarks.detach()

        for i in range(0, self.iteration):
            ROIs = 0
            # if phase == 'train':    # teacher forcing，下一次 ROI 位置是基于真值+噪声，而不是上一次的预测
            #     if i == 0:
            #         ROIs = labels + torch.from_numpy(np.random.normal(loc=0.0, scale=32.0 / self.origin_image_size[2] / 3, size = labels.size())).cuda(self.usegpu).float()
            #     elif i == 1:
            #         ROIs = labels + torch.from_numpy(np.random.normal(loc=0.0, scale=16.0 / self.origin_image_size[2] / 3, size = labels.size())).cuda(self.usegpu).float()
            #     else:
            #         ROIs = labels + torch.from_numpy(np.random.normal(loc=0.0, scale=8.0 / self.origin_image_size[2] / 3, size = labels.size())).cuda(self.usegpu).float()
            # else:
            #     ROIs = predict
            if phase == 'train':
                # 1. 计算噪声比例 (保持论文的多分辨率逻辑: 32 -> 16 -> 8)
                if i == 0:   scale_val = 32.0   # 对应论文里面的多分辨率，这里是半径
                elif i == 1: scale_val = 16.0
                else:        scale_val = 8.0
                
                # 生成噪声
                noise = torch.from_numpy(np.random.normal(
                    loc=0.0, 
                    scale=scale_val / self.origin_image_size[2] / 6,        # 噪声比例设置为半径的1/6
                    size=labels.size()
                )).cuda(self.usegpu).float()

                # 2. 确定切图中心 (ROIs)
                if i == 0:
                    # 第 0 步：冷启动，必须用 真值 + 噪声 (否则可能切到全黑)
                    ROIs = labels + noise
                else:
                    # 使用上一步的预测值 (predict) + 噪声  Student Forcing，强迫模型学会从上一步的位置修正
                    ROIs = predict.detach() + noise
            else:
                ROIs = predict  # 验证/测试阶段：始终使用上一步的预测

            ROIs = MyUtils.adjustment(ROIs, labels)
            # 这里加一个数值裁剪（限幅），避免ROIs越界
            ROIs = torch.clamp(ROIs, 0.0, 1.0)

            cropedtems = MyUtils.getcropedInputs_related(ROIs.detach().cpu().numpy(), labels, inputs_origin, -1, i, self.config)
            cropedtems = torch.cat([cropedtems[i].cuda(self.usegpu) for i in range(len(cropedtems))], dim=0)
            # 1. 获取 Encoder 的原始输出
            features_raw = self.encoder(cropedtems)     # 如果 crop_size=32，形状是 [B, 512, 1, 1, 1]；如果 crop_size=64，形状是 [B, 512, 2, 2, 2]  
            # 2. 强制压缩成 1x1x1
            features_pooled = torch.nn.functional.adaptive_avg_pool3d(features_raw, (1, 1, 1))  # 无论 crop_size 多大，这里输出形状永远是 [B, 512, 1, 1, 1]
            # 3. 调整维度以匹配后续全连接层
            features = features_pooled.view(features_pooled.size(0), -1).unsqueeze(0)   # 严谨写法：先展平为 [B, 512]，再 unsqueeze

            global_feature = MyUtils.get_global_feature(ROIs.detach().cpu().numpy(), coarse_feature, self.landmarkNum) # 获取全局特征（64维度）

            # 先拼接全局特征+局部特征，再GNN 的版本，测试后发现变化不大
            # features = torch.cat((features, global_feature), dim=2)     # 拼接: 512 + 64 = 576 维
            # features = self.graph_attention(ROIs, features)

            # 原始版本：先对全局特征GNN，再拼接局部特征
            global_feature = self.graph_attention(ROIs, global_feature)     # graph attention（GNN）
            features = torch.cat((features, global_feature), dim=2)
            # features = self.graph_attention(ROIs, features)

            # h_state = features
            # c_state = ROIs
            if i == 0:
                h_state = features
                c_state = ROIs
            else:
                gate_f = self.attention_gate_head(self.attention_gate_share(h_state.squeeze()).unsqueeze(0))
                gate_a = self.attention_gate_head(self.attention_gate_share(features.squeeze()).unsqueeze(0))
                gate = torch.softmax(torch.cat([gate_f, gate_a], dim=2), dim=2)

                h_state = h_state * gate[0, :, 0].view(1, -1, 1) + features * gate[0, :, 1].view(1, -1, 1)
                c_state = c_state * gate[0, :, 0].view(1, -1, 1) + ROIs * gate[0, :, 1].view(1, -1, 1)
                # c_state = ROIs

            x, y, z = self.decoders_offset_x(h_state), self.decoders_offset_y(h_state), self.decoders_offset_z(h_state)
            # print(size_tensor_inv)
            predict = torch.cat([x, y, z], dim=2) * size_tensor_inv + c_state
            predicts.append(predict.float())

        predicts = torch.cat(predicts, dim=0)

        return predicts # 返回的是所有迭代的结果

# --- 🔥 为了兼容新代码的 naming，添加别名 ---
CoarseNet = coarseNet
FineNet = fine_LSTM