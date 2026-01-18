import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse

# 引入项目模块
from MyModel import coarseNet
from MyDataLoader import LandmarksDataset, ToTensor
import MyUtils 

# ==========================================
# 配置参数
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--landmarkNum", type=int, default=7) 
parser.add_argument("--image_scale", default=(96, 96, 96), type=tuple) # (D, H, W)
parser.add_argument("--use_gpu", type=int, default=1)
parser.add_argument("--testcsv", type=str, default='test.csv') 
parser.add_argument("--data_enhanceNum", type=int, default=1) 
parser.add_argument("--spacing", default=(0.5, 0.5, 0.5), type=tuple)
parser.add_argument("--saveName", type=str, default='test3')   

def evaluate_all(config):
    print(f"🚀 开始全量评估 CoarseNet (权重: {config.saveName})...")
    device = torch.device("cuda" if config.use_gpu else "cpu")
    
    # 1. 准备数据
    transform_test = transforms.Compose([ToTensor()])
    dataset_path = "F:/CBCT/SA-LSTM-3D-Landmark-Detection2/processed_data/"
    
    test_dataset = LandmarksDataset(
        csv_file=dataset_path + config.testcsv,
        root_dir=dataset_path + "images",
        transform=transform_test,
        landmarksNum=config.landmarkNum
    )
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. 加载模型
    model = coarseNet(config).to(device)
    model_path = os.path.join('runs', config.saveName, 'best_coarse.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"✅ 模型加载成功: {model_path}")
    else:
        print(f"❌ 错误: 找不到权重文件 {model_path}")
        return

    model.eval()
    
    # 3. 构建 Global Coordinate (完全复刻 TrainNet)
    gl, gh, gw = config.image_scale
    global_coordinate = torch.ones(gl, gh, gw, 3).float()
    
    for i in range(gl):
        global_coordinate[i, :, :, 0] = global_coordinate[i, :, :, 0] * i
    for i in range(gh):
        global_coordinate[:, i, :, 1] = global_coordinate[:, i, :, 1] * i
    for i in range(gw):
        global_coordinate[:, :, i, 2] = global_coordinate[:, :, i, 2] * i
        
    global_coordinate = global_coordinate.to(device) * torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)]).to(device)

    all_errors_mm = [] 
    
    print("\n📊 正在逐个样本推理...")
    print(f"{'Sample ID':<20} | {'MRE (mm)':<15} | {'Physical Scale (mm)':<30}")
    print("-" * 80)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data['DICOM'].to(device)
            labels = data['landmarks'].to(device) # (B, N, 3) Truth [0-1]
            
            # --- 🔥 修正：计算物理尺寸 (mm) 🔥 ---
            try:
                # data['size'] 是 Tensor 形状 (Batch, 3) -> (1, 3)
                # 我们先取出第 0 个样本的数据
                size_data = data['size'][0] # 变成 Tensor([512, 512, 512])
                
                # 现在可以直接按索引 0, 1, 2 取值了
                # 假设 Dataset 返回顺序是 [Depth, Height, Width]
                pixel_z = size_data[0].item() # 512
                pixel_y = size_data[1].item() # 512
                pixel_x = size_data[2].item() # 512
                
                # 2. 获取 Spacing [z, y, x]
                sp_z, sp_y, sp_x = config.spacing # (0.4, 0.4, 0.4)
                
                # 3. 构建物理 Scale 向量 (mm)
                scale_w = pixel_x * sp_x # Width (mm)
                scale_h = pixel_y * sp_y # Height (mm)
                scale_d = pixel_z * sp_z # Depth (mm)
                
                # ⚠️ 注意这里顺序: [Width, Height, Depth]
                physical_scale = np.array([scale_w, scale_h, scale_d])
                
            except Exception as e:
                print(f"⚠️ Scale 计算失败: {e}, 使用默认 204.8mm")
                physical_scale = np.array([204.8, 204.8, 204.8])

            # A. 推理
            pred_heatmaps_list, _ = model(inputs)
            
            # B. 计算坐标
            # pred_coords: (1, N, 3) -> [x, y, z] (MyUtils已修复)
            pred_coords = MyUtils.get_coordinates_from_coarse_heatmaps(pred_heatmaps_list, global_coordinate)
            
            # C. 计算误差
            # 过滤无效点
            mask = (labels[:, :, 0] >= 0).cpu().numpy()
            
            diff = torch.abs(pred_coords - labels).cpu().numpy() # (1, N, 3) [0-1]
            
            # 还原物理尺寸 (mm)
            # 误差 = 归一化差值 * 物理全尺寸(mm)
            diff_mm = diff * physical_scale
            
            # 计算欧氏距离
            dist_mm = np.linalg.norm(diff_mm, axis=2) # (1, N)
            
            # 只统计有效点
            valid_dists = dist_mm[mask]
            
            all_errors_mm.extend(valid_dists)
            
            # 打印当前样本均值
            sample_mre = np.mean(valid_dists) if len(valid_dists) > 0 else 0
            
            sample_name = data['imageName'][0]
            
            # 格式化 scale 字符串方便检查
            scale_str = f"[{physical_scale[0]:.1f}, {physical_scale[1]:.1f}, {physical_scale[2]:.1f}]"
            
            print(f"{sample_name[:20]:<20} | {sample_mre:<15.4f} | {scale_str:<30}")

    # 4. 最终汇总
    print("\n" + "="*50)
    print("📈 全量评估报告 (Coarse Stage Only)")
    print("="*50)
    
    if len(all_errors_mm) == 0:
        print("没有找到任何有效关键点！")
        return

    mre = np.mean(all_errors_mm)
    sd = np.std(all_errors_mm)
    max_e = np.max(all_errors_mm)
    
    print(f"Total Valid Landmarks : {len(all_errors_mm)}")
    print("-" * 30)
    print(f"MRE (Mean Radial Error): {mre:.4f} mm")
    print(f"SD  (Standard Deviation): {sd:.4f} mm")
    print(f"Max Error               : {max_e:.4f} mm")
    print("="*50)

if __name__ == '__main__':
    config = parser.parse_args()
    evaluate_all(config)