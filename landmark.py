from __future__ import print_function, division
import torch
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from MyDataLoader import ToTensor, LandmarksDataset
import MyModel
import TrainNet
import LossFunction
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # 忽略所有 UserWarning 类型的警告
plt.ion()  # interactive mode

parser = argparse.ArgumentParser()
# 模型训练部分的参数
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--landmarkNum", type=int, default=7)
parser.add_argument("--image_scale", default=(96, 96, 96), type=tuple)  # 降采样图像尺寸
parser.add_argument("--origin_image_size", default=(512, 512, 512), type=tuple) # 原始图像尺寸
parser.add_argument("--crop_size", default=(64, 64, 64), type=tuple)        # 裁剪块尺寸, 32修改为64
parser.add_argument("--use_gpu", type=int, default=0)
parser.add_argument("--iteration", type=int, default=3)                 # LSTM的长度
# parser.add_argument("--R1", type=int, default=5)              # 暂时没有使用到这两个参数
# parser.add_argument("--R2", type=int, default=9)
parser.add_argument("--epochs", type=int, default=50)          # 迭代次数
parser.add_argument("--data_enhanceNum", type=int, default=5)   # TODO:数据增强
parser.add_argument('--lr', type=float, default=0.0001)     # 学习率
parser.add_argument("--spacing", type=tuple, default=(0.5, 0.5, 0.5))   # npy数据的体素间距
parser.add_argument("--stage", type=str, default="train")       # 默认为训练模式
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g., 'runs/exp1')") # 从哪个路径下的权重开始继续训练
# 输入数据部分参数
parser.add_argument('--dataRoot', type=str, default="./processed_data/")   # npy格式数据路径  使用根目录，这样服务器上面可以直接运行
parser.add_argument("--traincsv", type=str, default='train.csv')    # 训练数据
parser.add_argument("--testcsv", type=str, default='test.csv')      # 测试数据
# 输出保存部分参数 
parser.add_argument("--saveName", type=str, default='Data_enhance_test')         # 修改配置以后要修改saveName来保存训练数据
# 加载哪个文件夹的权重进行测试
parser.add_argument("--testName", type=str, default="test3")    # 选择哪个配置来测试数据


def main():
    config = parser.parse_args()
    fine_LSTM = MyModel.fine_LSTM(config).cuda(config.use_gpu)
    coarseNet = MyModel.coarseNet(config).cuda(config.use_gpu)

    # 定义数据预处理流水线(Pipeline)转换为Tensor格式
    transform_origin = transforms.Compose([
        # Rescale(config.origin_image_size),    # 图像在预处理的时候已经Resize了
        ToTensor()
    ])

    # 测试模式
    if config.stage == 'test':
        print(f"🚀 Mode: TEST | Loading weights from: {config.testName}")
        
        # 加载权重
        save_dir = os.path.join('runs', config.testName)
        coarseNet.load_state_dict(torch.load(os.path.join(save_dir, 'best_coarse.pth')))
        fine_LSTM.load_state_dict(torch.load(os.path.join(save_dir, 'best_fine_LSTM.pth')))
            
        # 准备测试数据
        test_dataset = LandmarksDataset(
            csv_file=config.dataRoot + config.testcsv,
            root_dir=config.dataRoot + "images",
            transform=transform_origin,
            landmarksNum=config.landmarkNum,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # 3. 执行测试
        TrainNet.test_model(coarseNet, fine_LSTM, test_dataloader, config)
        return # 测试结束后直接退出

    train_dataset_origin = LandmarksDataset(csv_file=config.dataRoot + config.traincsv,
                                            root_dir=config.dataRoot + "images",
                                            transform=transform_origin,
                                            landmarksNum=config.landmarkNum
                                            )

    val_dataset = LandmarksDataset(csv_file=config.dataRoot + config.testcsv,
                                   root_dir=config.dataRoot + "images",
                                   transform=transform_origin,
                                   landmarksNum=config.landmarkNum
                                   )
    
    # train_dataloader = []
    # val_dataloader = []

    # # 创建训练数据加载器，可高效读取的批量数据
    # train_dataloader_t = DataLoader(train_dataset_origin, batch_size=config.batchSize, shuffle=False, num_workers=0)
    # for data in train_dataloader_t:
    #     train_dataloader.append(data)

    # val_dataloader_t = DataLoader(val_dataset, batch_size=config.batchSize, shuffle=False, num_workers=0)
    # for data in val_dataloader_t:
    #     val_dataloader.append(data)

    # 直接使用 DataLoader，不要用 List Append！
    # 这样程序会瞬间启动，训练时后台加载
    train_dataloader = DataLoader(
        train_dataset_origin, 
        batch_size=config.batchSize, 
        shuffle=True,       # 必须打乱
        num_workers=0,      # 设置为 CPU 核心数
        pin_memory=True,    # 加速 CPU -> GPU 传输
        # persistent_workers=True
    )
    # 验证集
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.batchSize, 
        shuffle=False, 
        num_workers=0,
        # pin_memory=True
    )

    print(f"train data:{len(train_dataloader)}, test data:{len(val_dataloader)}")

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    criterion_coarse = LossFunction.coarse_heatmap(config)
    criterion_fine = LossFunction.fine_heatmap(config)

    params = list(coarseNet.parameters()) + list(fine_LSTM.parameters())

    optimizer_ft = optim.Adam(params, lr=config.lr, weight_decay=1e-4)  # 权重衰减，如果依然过拟合，尝试加重到1e-3

    TrainNet.train_model(coarseNet, fine_LSTM, dataloaders, criterion_coarse, criterion_fine,
                         optimizer_ft, config)

if __name__ == "__main__":
    main()