import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

# 引入项目模块
from MyModel import coarseNet
from MyDataLoader import LandmarksDataset, ToTensor
import LossFunction 
import MyUtils # 🔥 直接调用 MyUtils

# ==========================================
# 配置参数 (保持与训练一致)
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--landmarkNum", type=int, default=7)
parser.add_argument("--image_scale", default=(96, 96, 96), type=tuple) # (D, H, W)
parser.add_argument("--origin_image_size", default=(512, 512, 512), type=tuple)
parser.add_argument("--use_gpu", type=int, default=1)
parser.add_argument("--testcsv", type=str, default='test.csv')
parser.add_argument("--saveName", type=str, default='test2') # ⚠️ 确保这里是你训练好的权重文件夹
parser.add_argument("--data_enhanceNum", type=int, default=1)
parser.add_argument("--stage", type=str, default="train")

# -----------------------------------------------------------------------------
# 辅助函数：复刻 TrainNet.py 中的全局坐标构建逻辑
# -----------------------------------------------------------------------------
def build_global_coordinate(config, device):
    """
    必须与 TrainNet.py 中的构建逻辑 100% 保持一致！
    """
    gl, gh, gw = config.image_scale
    global_coordinate = torch.ones(gl, gh, gw, 3).float()
    
    # 按照 TrainNet 的逻辑构建
    for i in range(gl):
        global_coordinate[i, :, :, 0] = global_coordinate[i, :, :, 0] * i # Ch0: Z (Depth)
    for i in range(gh):
        global_coordinate[:, i, :, 1] = global_coordinate[:, i, :, 1] * i # Ch1: Y (Height)
    for i in range(gw):
        global_coordinate[:, :, i, 2] = global_coordinate[:, :, i, 2] * i # Ch2: X (Width)
        
    # 归一化
    scale_factor = torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)])
    global_coordinate = global_coordinate * scale_factor
    
    return global_coordinate.to(device)

# -----------------------------------------------------------------------------
# 主调试函数
# -----------------------------------------------------------------------------
def debug_visualization(config):
    print("🚀 开始 Debug: 调用项目内部函数进行验证 ...")
    device = torch.device("cuda" if config.use_gpu else "cpu")
    
    # 1. 实例化 LossFunction (用于生成 GT 热图做对比)
    criterion = LossFunction.coarse_heatmap(config)
    
    # 2. 准备数据
    transform_test = transforms.Compose([ToTensor()])
    test_dataset = LandmarksDataset(
        csv_file="F:/CBCT/SA-LSTM-3D-Landmark-Detection2/processed_data/" + config.testcsv,
        root_dir="F:/CBCT/SA-LSTM-3D-Landmark-Detection2/processed_data/" + "images",
        transform=transform_test,
        landmarksNum=config.landmarkNum
    )
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 取一个样本
    data = next(iter(dataloader))
    inputs = data['DICOM'].to(device)
    labels = data['landmarks'].to(device) # (B, N, 3) Truth
    
    # ❌ [删除或注释这一行] 因为 dataset 里没有 'name' 这个 key
    # print(f"📖 加载样本: {data['name'][0]}") 
    print("📖 样本加载成功")

    # 3. 加载模型
    model = coarseNet(config).to(device)
    model_path = os.path.join('runs', config.saveName, 'best_coarse.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"✅ 模型权重已加载: {model_path}")
    else:
        print(f"❌ 未找到权重: {model_path}，使用随机初始化 (预测将不准确)")
    
    model.eval()
    
    # 4. 推理 & 计算坐标
    with torch.no_grad():
        # A. 模型推理 -> 得到 Heatmaps
        pred_heatmaps_list, _ = model(inputs)
        
        # B. 构建全局坐标系 (复刻 TrainNet)
        global_coordinate = build_global_coordinate(config, device)
        
        # C. 🔥 [核心] 调用 MyUtils 计算坐标 🔥
        # MyUtils 期望传入的是 list of tensors，以及 global_coordinate
        pred_coords = MyUtils.get_coordinates_from_coarse_heatmaps(pred_heatmaps_list, global_coordinate)
        # pred_coords shape: (N, 3) - 注意这里 MyUtils 应该返回 N 个点的坐标
    
    # 5. 分析结果 (只看第 1 个关键点，Index 1)
    target_idx = 1
    
    label_np = labels.cpu().numpy()[0, target_idx] # Truth [x, y, z]
    pred_np = pred_coords.cpu().numpy()[target_idx] # Predict [x, y, z] (if fixed)
    
    print("\n" + "="*50)
    print("📊 坐标精度验证 (Coordinate Accuracy Check)")
    print(f"   关键点 Index: {target_idx}")
    print("-" * 30)
    print(f"   Label (Truth)   : {label_np}")
    print(f"   MyUtils Predict : {pred_np}")
    
    diff = np.abs(label_np - pred_np)
    mre = np.linalg.norm(diff)
    
    print("-" * 30)
    print(f"   Diff (Abs)      : {diff}")
    print(f"   MRE (Normalized): {mre:.4f}")
    
    # 将归一化误差转换为物理距离 (假设图像 96mm / 96px, 1.0=96mm)
    print(f"   MRE (Approx mm) : {mre * 96:.2f} mm (假设 Scale=96)")
    
    if mre < 0.05:
        print("\n✅ [PASS] MyUtils 计算正确！误差极小。")
    else:
        print("\n❌ [FAIL] 误差依然很大，请检查 MyUtils 索引修复是否生效。")
    print("="*50 + "\n")

    # 6. 可视化 (Heatmap 层面再次确认)
    scale = torch.tensor([config.image_scale[2]-1, config.image_scale[1]-1, config.image_scale[0]-1], device=device)
    labels_pixel = labels * scale
    gt_heatmap_tensor = criterion.generate_target_heatmap(labels_pixel, 1, device) # (1, N, D, H, W)
    gt_map = gt_heatmap_tensor[0, target_idx].cpu().numpy()
    
    # 获取预测热图
    pred_map = pred_heatmaps_list[target_idx][0].cpu().numpy()
    
    # 找最大值切片
    z_slice = np.argmax(gt_map) // (gt_map.shape[1] * gt_map.shape[2])
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"GT Heatmap (Label)\nSlice Z={z_slice}")
    plt.imshow(gt_map[z_slice], cmap='jet')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title(f"Model Prediction\nSlice Z={z_slice}")
    plt.imshow(pred_map[z_slice], cmap='jet')
    plt.colorbar()
    
    plt.savefig('debug_final_check.png')
    print("🖼️ 热图对比已保存为 debug_final_check.png")

if __name__ == '__main__':
    config = parser.parse_args()
    debug_visualization(config)