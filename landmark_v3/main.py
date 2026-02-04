import os
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import warnings

# --- 引用重构后的模块 ---
from config import Config
from models import legacy_models as MyModel
from data import legacy_dataset as MyDataLoader
from core import legacy_loss as LossFunction
from core.legacy_trainer import Trainer

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # 1. 加载配置
    config = Config().parse()
    
    print(f"🚀 Start Mode: {config.stage.upper()}")
    
    # 2. 初始化模型
    # 注意：旧代码里是 MyModel.fine_LSTM(config) 和 MyModel.coarseNet(config)
    fine_LSTM = MyModel.fine_LSTM(config).cuda(config.use_gpu)
    coarseNet = MyModel.coarseNet(config).cuda(config.use_gpu)

    # 3. 数据预处理
    transform_origin = transforms.Compose([
        MyDataLoader.ToTensor()
    ])

    # --- TEST 模式 ---
    if config.stage == 'test':
        print(f"🚀 Loading weights from: {config.testName}")
        save_dir = os.path.join('runs', config.testName)
        
        # 加载权重
        coarse_path = os.path.join(save_dir, 'best_coarse.pth')
        fine_path = os.path.join(save_dir, 'best_fine_LSTM.pth')
        
        if os.path.exists(coarse_path) and os.path.exists(fine_path):
            coarseNet.load_state_dict(torch.load(coarse_path))
            fine_LSTM.load_state_dict(torch.load(fine_path))
        else:
            print(f"❌ Error: Weights not found in {save_dir}")
            return

        # 准备测试数据
        test_dataset = MyDataLoader.LandmarksDataset(
            csv_file=config.dataRoot + config.testcsv,
            root_dir=config.dataRoot + "images",
            transform=transform_origin,
            landmarksNum=config.landmarkNum,
        )
        # 测试用 DataLoader (旧代码这里没有转 list，直接用的 DataLoader)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # 初始化 Trainer (为了调用 test 方法)
        # 测试模式下 loss 和 optimizer 可以为空
        trainer = Trainer(config, coarseNet, fine_LSTM, {'val': test_dataloader}, None, None, None)
        trainer.test()
        return

    # --- TRAIN 模式 ---
    
    # 4. 准备数据集
    train_dataset = MyDataLoader.LandmarksDataset(
        csv_file=config.dataRoot + config.traincsv,
        root_dir=config.dataRoot + "images",
        transform=transform_origin,
        landmarksNum=config.landmarkNum
    )

    val_dataset = MyDataLoader.LandmarksDataset(
        csv_file=config.dataRoot + config.testcsv,
        root_dir=config.dataRoot + "images",
        transform=transform_origin,
        landmarksNum=config.landmarkNum
    )
    
    # 5. 🔥 复刻旧逻辑：把 DataLoader 转为 List (虽然很耗内存，但为了保持一致)
    # 旧代码:
    # train_dataloader_t = DataLoader(..., shuffle=False)
    # for data in train_dataloader_t: train_dataloader.append(data)
    
    train_loader_raw = DataLoader(train_dataset, batch_size=config.batchSize, shuffle=False, num_workers=0)
    val_loader_raw = DataLoader(val_dataset, batch_size=config.batchSize, shuffle=False, num_workers=0)
    
    train_data_list = []
    print("⏳ Pre-loading Training Data into RAM (Legacy Mode)...")
    for data in train_loader_raw:
        train_data_list.append(data)
        
    val_data_list = []
    print("⏳ Pre-loading Validation Data into RAM (Legacy Mode)...")
    for data in val_loader_raw:
        val_data_list.append(data)
        
    print(f"✅ Loaded: Train {len(train_data_list)}, Val {len(val_data_list)}")
    
    dataloaders = {'train': train_data_list, 'val': val_data_list}

    # 6. 初始化 Loss 和 Optimizer
    criterion_coarse = LossFunction.coarse_heatmap(config).cuda(config.use_gpu)
    criterion_fine = LossFunction.fine_heatmap(config).cuda(config.use_gpu) # 这里的fine_heatmap可能没用上，因为旧 Trainer 里手写了 SmoothL1，但为了兼容先传进去

    params = list(coarseNet.parameters()) + list(fine_LSTM.parameters())
    optimizer = optim.AdamW(params, lr=config.lr, weight_decay=5e-4)

    # 7. 启动 Trainer
    trainer = Trainer(config, coarseNet, fine_LSTM, dataloaders, criterion_coarse, criterion_fine, optimizer)
    trainer.run()

if __name__ == "__main__":
    main()