from __future__ import print_function, division
import torch
import numpy as np
import time
import MyUtils
import torch.nn.functional as F
import processData
import LossFunction
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

def train_model(coarse_net, fine_LSTM, dataloaders, criterion_coarse, criterion_fine, optimizer, config):
    since = time.time()
    test_epoch = 1         # epoch为5的倍数的时候，验证模型在测试集上的效果

    best_mre = float('inf')     # 最佳MRE

    # 初始化 SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join('runs', config.saveName))

    # 记录配置参数
    config_str = " | Parameter | Value |\n|---|---|\n"
    for key, value in vars(config).items():
        config_str += f"| {key} | {str(value)} |\n"
    writer.add_text('Experiment_Config', config_str, 0)

    # 保存参数到 txt
    if not os.path.exists(os.path.join('runs', config.saveName)):
        os.makedirs(os.path.join('runs', config.saveName))
        
    with open(os.path.join(os.path.join('runs', config.saveName), 'config.txt'), 'w') as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")

    # --- 准备全局坐标网格 ---
    gl, gh, gw = config.image_scale
    global_coordinate = torch.ones(gl, gh, gw, 3).float()
    for i in range(gl):
        global_coordinate[i, :, :, 0] = global_coordinate[i, :, :, 0] * i
    for i in range(gh):
        global_coordinate[:, i, :, 1] = global_coordinate[:, i, :, 1] * i
    for i in range(gw):
        global_coordinate[:, :, i, 2] = global_coordinate[:, :, i, 2] * i
    global_coordinate = global_coordinate.cuda(config.use_gpu) * torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)]).cuda(config.use_gpu)

    # --- 训练循环 ---
    for epoch in range(config.epochs):

        train_fine_Off = []
        train_fine_Off_heatmap = []
        train_coarse_Off = []
        test_fine_Off = []
        test_fine_Off_heatmap = []
        test_coarse_Off = []

        for phase in ['train', 'val']:
            # 直接获取 DataLoader
            datas = dataloaders[phase]
            
            # 手动管理 tqdm，保持原始风格
            pbar = tqdm(total=len(datas), desc=f'{phase} Epoch {epoch}')

            if phase == 'train':
                if config.stage == 'test': continue
                coarse_net.train(True) 
                fine_LSTM.train(True)
            else:
                if epoch % test_epoch != 0:
                    continue
                coarse_net.train(False) 
                fine_LSTM.train(False)

            lent = len(datas)
            running_loss = 0

            # 遍历数据
            for data in datas:
                inputs = data['DICOM'].cuda(config.use_gpu) # (B, C, D, H, W)
                
                inputs_origin_list = data['DICOM_origin']
                inputs_origin = [item.squeeze(0) for item in inputs_origin_list]

                labels = data['landmarks'].cuda(config.use_gpu) 
                
                size = data['size'][0]
                
                # 构造物理尺寸张量
                size_tensor = torch.tensor([size[1], size[2], size[0]]).float().cuda(config.use_gpu).unsqueeze(0)
                size_tensor_inv = 1.0 / size_tensor.float()
                
                optimizer.zero_grad()

                # 显存控制：验证阶段不构建计算图 防止验证集吃掉显存
                with torch.set_grad_enabled(phase == 'train'):
                    # 前向传播 (Forward)
                    coarse_heatmap, coarse_feature = coarse_net(inputs)

                    # 第一道关卡：检查网络输出是否正常 
                    has_nan = any(torch.isnan(h).any() for h in coarse_heatmap)     # 检查 list 中任何一个 tensor 是否有 NaN
                    if has_nan:     # 如果 CoarseNet 输出里就有 NaN，说明网络内部炸了 此时必须跳过，不能把 NaN 传给 MyUtils，否则会爆内存
                        print(f"⚠️ [Warning] NaN detected in CoarseNet output at Epoch {epoch}. Skipping this batch.")
                        optimizer.zero_grad() # 清空梯度
                        continue # 🔥 直接跳过！不跑 FineNet，不反向传播
                    
                    # 获取粗定位坐标
                    coarse_landmarks = MyUtils.get_coordinates_from_coarse_heatmaps(coarse_heatmap, global_coordinate).unsqueeze(0)
                    # 强制限制坐标在 0-1 ，防止预测跑出边界
                    coarse_landmarks = torch.clamp(coarse_landmarks, 0.0, 1.0)
                    
                    # Fine Stage
                    fine_landmarks = fine_LSTM(coarse_landmarks, labels, inputs_origin, coarse_feature, phase, size_tensor_inv)

                    # 计算 Loss (Original Logic)
                    mask = (labels[:, :, 0] >= 0).float().unsqueeze(2)
                    loss = (torch.abs(fine_landmarks - labels) * mask).sum() / (mask.sum() * 3 + 1e-6)
                    
                    # Coarse Loss: 传入 List 类型的 coarse_heatmap
                    loss += criterion_coarse(coarse_heatmap, global_coordinate, labels, phase)

                    # 反向传播
                    if phase == 'train' and config.stage == 'train':
                        # 第二道关卡：检查 Loss 是否正常
                        if torch.isnan(loss):
                            print(f"⚠️ [Warning] Loss is NaN at Epoch {epoch}. Skipping gradient update.")
                            optimizer.zero_grad()
                            continue # 🔥 跳过更新

                        loss.backward()
                        # 梯度裁剪 (防止梯度爆炸导致下一次预测飞出天际)
                        torch.nn.utils.clip_grad_norm_(coarse_net.parameters(), max_norm=5.0)
                        torch.nn.utils.clip_grad_norm_(fine_LSTM.parameters(), max_norm=5.0)
                        optimizer.step()

                # -------------------------------------------------------------------
                # 4. 指标统计
                # -------------------------------------------------------------------
                if epoch % test_epoch == 0:
                    coarse_off = MyUtils.get_coarse_errors(coarse_landmarks, global_coordinate, labels, size_tensor)
                    fine_off = MyUtils.get_fine_errors(fine_landmarks, labels, size_tensor)

                    fine_off_heatmap = coarse_off 
                    
                    if phase == "train":
                        train_fine_Off.append(fine_off.detach().cpu())
                        train_fine_Off_heatmap.append(fine_off_heatmap.detach().cpu())
                        train_coarse_Off.append(coarse_off.detach().cpu())
                    else:
                        test_fine_Off.append(fine_off.detach().cpu())
                        test_fine_Off_heatmap.append(fine_off_heatmap.detach().cpu())
                        test_coarse_Off.append(coarse_off.detach().cpu())
                
                running_loss += loss.item()
                pbar.update(1)

            # End of Epoch
            epoch_loss = running_loss / lent
            pbar.close()
            
            if epoch % 1 == 0:
                print('{} epoch: {} Loss: {:.4f}'.format(phase, epoch, epoch_loss))
            
            if phase == 'train':
                writer.add_scalar('Loss/Train', epoch_loss, epoch)
            elif phase == 'val':
                writer.add_scalar('Loss/Val', epoch_loss, epoch)

        # -------------------------------------------------------------------
        # 5. TensorBoard 记录与结果保存 (保持原始逻辑)
        # -------------------------------------------------------------------
        if epoch % test_epoch == 0:
            current_val_mre = float('inf') # 用于记录当前轮次的验证集 MRE
            if len(test_fine_Off) > 0:
                test_fine_Off = torch.cat(test_fine_Off, dim=0)
                test_fine_Off_heatmap = torch.cat(test_fine_Off_heatmap, dim=0)
                test_coarse_Off = torch.cat(test_coarse_Off, dim=0)

                test_coarse_SDR, test_coarse_SD, test_coarse_MRE = MyUtils.analysis_result(config.landmarkNum, test_coarse_Off.numpy())
                test_fine_SDR, test_fine_SD, test_fine_MRE = MyUtils.analysis_result(config.landmarkNum, test_fine_Off.numpy())
                
                # 记录 Test 结果
                writer.add_scalars('Comparison/MRE_Test', {'Coarse': np.mean(test_coarse_MRE), 'Fine': np.mean(test_fine_MRE)}, epoch)
                writer.add_scalar('Test/Fine_MRE', np.mean(test_fine_MRE), epoch)
                writer.add_scalar('Test/Coarse_MRE', np.mean(test_coarse_MRE), epoch)
                writer.add_scalar('Test/Fine_SD', np.mean(test_fine_SD), epoch)
                writer.add_scalar('Test/CoarseSD', np.mean(test_coarse_SD), epoch)
                
                current_val_mre = np.mean(test_fine_MRE) # 获取当前 Fine Stage 的 MRE
                test_sdr_mean = np.mean(test_fine_SDR, axis=0)
                writer.add_scalar('Test_SDR/2.0mm', test_sdr_mean[1], epoch)
                writer.add_scalar('Test_SDR/4.0mm', test_sdr_mean[3], epoch)
                writer.add_scalar('Test_SDR/6.0mm', test_sdr_mean[5], epoch)
                writer.add_scalar('Test_SDR/8.0mm', test_sdr_mean[7], epoch)

                if (epoch + 1) % 10 == 0:
                    # 打印 Test 结果
                    print(f"\n [TEST SET Results]")
                    print(f"   Fine Stage   -> MRE: {np.mean(test_fine_MRE):.4f} mm | SD: {np.mean(test_fine_SD):.4f} mm")
                    print(f"   Coarse Stage -> MRE: {np.mean(test_coarse_MRE):.4f} mm | SD: {np.mean(test_coarse_SD):.4f} mm")
                    print(f"   SDR (Thresholds):")
                    print(f"     2.0mm: {test_sdr_mean[1]:.2f}%")
                    print(f"     4.0mm: {test_sdr_mean[3]:.2f}%")
                    print(f"     6.0mm: {test_sdr_mean[5]:.2f}%")
                    print(f"     8.0mm: {test_sdr_mean[7]:.2f}%") 
                    print(f"     Full SDR Vector: {np.round(test_sdr_mean, 2)}")

            if config.stage == 'train' and len(train_fine_Off) > 0:
                train_fine_Off = torch.cat(train_fine_Off, dim=0)
                train_fine_Off_heatmap = torch.cat(train_fine_Off_heatmap, dim=0)
                train_coarse_Off = torch.cat(train_coarse_Off, dim=0)

                train_coarse_SDR, train_coarse_SD, train_coarse_MRE = MyUtils.analysis_result(config.landmarkNum, train_coarse_Off.numpy())
                train_fine_SDR, train_fine_SD, train_fine_MRE = MyUtils.analysis_result(config.landmarkNum, train_fine_Off.numpy())

                # 记录 Train 结果
                writer.add_scalars('Comparison/MRE_Train', {'Coarse': np.mean(train_coarse_MRE), 'Fine': np.mean(train_fine_MRE)}, epoch)
                writer.add_scalar('Train/Fine_MRE', np.mean(train_fine_MRE), epoch)
                writer.add_scalar('Train/Coarse_MRE', np.mean(train_coarse_MRE), epoch)
                writer.add_scalar('Train/Fine_SD', np.mean(train_fine_SD), epoch)
                writer.add_scalar('Train/Coarse_SD', np.mean(train_coarse_SD), epoch)

                
                train_sdr_mean = np.mean(train_fine_SDR, axis=0)
                writer.add_scalar('Train_SDR/2.0mm', train_sdr_mean[1], epoch)
                writer.add_scalar('Train_SDR/4.0mm', train_sdr_mean[3], epoch)
                writer.add_scalar('Train_SDR/6.0mm', train_sdr_mean[5], epoch)
                writer.add_scalar('Train_SDR/8.0mm', train_sdr_mean[7], epoch)

                if (epoch + 1) % 10 == 0:
                    # 打印 Train 结果
                    print(f"\n [TRAIN SET Results]")
                    print(f"   Fine Stage   -> MRE: {np.mean(train_fine_MRE):.4f} mm | SD: {np.mean(train_fine_SD):.4f} mm")
                    print(f"   Coarse Stage -> MRE: {np.mean(train_coarse_MRE):.4f} mm | SD: {np.mean(train_coarse_SD):.4f} mm")
                    print(f"   SDR (Thresholds):")
                    print(f"     2.0mm: {train_sdr_mean[1]:.2f}%")
                    print(f"     4.0mm: {train_sdr_mean[3]:.2f}%")
                    print(f"     6.0mm: {train_sdr_mean[5]:.2f}%")
                    print(f"     8.0mm: {train_sdr_mean[7]:.2f}%")
                    print(f"     Full SDR Vector: {np.round(train_sdr_mean, 2)}")
                    print()
            
            save_dir = os.path.join('runs', config.saveName)
            # 保存模型
            if current_val_mre < best_mre:
                print(f"🔥 New Best Model Found! MRE improved from {best_mre:.4f} to {current_val_mre:.4f}")
                best_mre = current_val_mre
                
                # 保存权重
                torch.save(coarse_net.state_dict(), os.path.join(save_dir, 'best_coarse.pth'))
                torch.save(fine_LSTM.state_dict(), os.path.join(save_dir, 'best_fine_LSTM.pth'))
                
                # 记录最佳 MRE 到 txt 文件，方便快速查看
                with open(os.path.join(save_dir, 'best_mre_record.txt'), 'w') as f:
                    f.write(f"Best MRE: {best_mre:.4f} at Epoch {epoch}\n")
            
            # 2. 保存最新模型 (Latest Model) - 防止断电白跑，每轮验证都覆盖保存
            torch.save(coarse_net.state_dict(), os.path.join(save_dir, 'latest_coarse.pth'))
            torch.save(fine_LSTM.state_dict(), os.path.join(save_dir, 'latest_fine_LSTM.pth'))

        print()
        # 强制清空显存缓存  解决 "有空闲但无法分配" 的碎片化问题
        torch.cuda.empty_cache()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    writer.close()


def test_model(corseNet, fineNet, dataloaders, criterion1, criterion2, optimizer, config):
    since = time.time()

    # num_epochs, use_gpu, R1, R2, saveName, landmarkNum, image_scale

    best_acc = [0, 0, 0, 0, 0, 0]
    test_avgOff = 0
    # Each epoch has a training and validation phase
    phase = 'val'
    corseNet.train(False)  # Set model to evaluate mode
    fineNet.train(False)

    # Iterate over data.
    lent = len(dataloaders[phase])
    test_Off = np.zeros((0, config.landmarkNum))
    for ide in range(lent):
        data = dataloaders[phase][ide]
        inputs, inputs_origin, labels = data['DICOM'].cuda(config.use_gpu), data['DICOM_origin'], data['landmarks']
        optimizer.zero_grad()

        heatMapsCorse, coordinatesCorse = corseNet(inputs)

        coordinatesCorse = coordinatesCorse.unsqueeze(0)

        ROIs = coordinatesCorse.cpu().detach().numpy()

        cropedtem = MyUtils.getcropedInputs(ROIs, inputs_origin, 64, -1)
        cropedInputs = [cropedInput.cuda(config.use_gpu) for cropedInput in cropedtem]
        data['cropedInputs'] = cropedInputs

        cropedInputs = data['cropedInputs']
        outputs2 = 0
        heatMapsFine, coordinatesFine = fineNet(ROIs, cropedInputs, outputs2)
        coordinatesFine = coordinatesFine.unsqueeze(0)
        print(coordinatesFine)
        coorall = coordinatesCorse + coordinatesFine
        # coorall = coordinatesCorse
        off = MyUtils.getCoordinate_new(heatMapsCorse, heatMapsFine, labels, config.R1, config.R2, config.use_gpu, coordinatesCorse,
                                        coordinatesFine, config)  # getCoordinate_new1

        print(off)

    time_elapsed = time.time() - since
    print('test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
