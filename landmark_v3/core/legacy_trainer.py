import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- 🔥 引用修正 ---
from utils import legacy_utils as MyUtils
from core import legacy_loss as LossFunction

class Trainer:
    def __init__(self, config, coarse_net, fine_LSTM, dataloaders, criterion_coarse, criterion_fine, optimizer):
        self.config = config
        self.coarse_net = coarse_net
        self.fine_LSTM = fine_LSTM
        self.dataloaders = dataloaders
        self.criterion_coarse = criterion_coarse
        self.criterion_fine = criterion_fine
        self.optimizer = optimizer

    def run(self):
        """
        对应旧代码中的 train_model 函数
        """
        # 为了不改动下面的逻辑代码，我们把 self.xxx 赋值给局部变量
        # 这样下面的代码几乎可以原样粘贴
        config = self.config
        coarse_net = self.coarse_net
        fine_LSTM = self.fine_LSTM
        dataloaders = self.dataloaders
        criterion_coarse = self.criterion_coarse
        criterion_fine = self.criterion_fine
        optimizer = self.optimizer
        
        since = time.time()
        test_epoch = 1         # epoch为5的倍数的时候，验证模型在测试集上的效果
        best_mre = float('inf')     # 最佳MRE
        start_epoch = 0         # 默认从 epoch0 开始

        # --- 1. 准备保存路径 ---
        save_dir = os.path.join('runs', config.saveName)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # --- 2. 初始化 Logger (关键修改) ---
        # 日志文件将保存在 runs/你的实验名/train.log
        log_path = os.path.join(save_dir, 'train.log')
        # 调用我们在 MyUtils 里写的函数
        logger = MyUtils.get_logger(log_path) 
        logger.info(f"🚀 Start Training: {config.saveName}")
        logger.info(f"📁 Logs and weights will be saved to: {save_dir}")
        logger.info("")

        # 初始化 SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join('runs', config.saveName))
        # tensorboard记录配置参数
        config_str = " | Parameter | Value |\n|---|---|\n"
        for key, value in vars(config).items():
            config_str += f"| {key} | {str(value)} |\n"
        writer.add_text('Experiment_Config', config_str, 0)

        # 配置参数也记录到 logger
        logger.info("------ Experiment Configuration ------")
        for key, value in vars(config).items():
            logger.info(f"{key}: {value}")
        logger.info("--------------------------------------")
        logger.info("")

        # --- 准备全局坐标网格 ---
        gl, gh, gw = config.image_scale
        global_coordinate = torch.ones(gl, gh, gw, 3).float()
        for i in range(gl): global_coordinate[i, :, :, 0] *= i
        for i in range(gh): global_coordinate[:, i, :, 1] *= i
        for i in range(gw): global_coordinate[:, :, i, 2] *= i
        global_coordinate = global_coordinate.cuda(config.use_gpu) * torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)]).cuda(config.use_gpu)

        # --- 断点加载逻辑 (Resume) ---
        if config.resume:
            checkpoint_path = config.resume
            if os.path.isdir(checkpoint_path):   # 如果传入的是文件夹 (例如: runs/test4)，自动拼接文件名
                checkpoint_path = os.path.join(checkpoint_path, 'latest_checkpoint.pth')
            
            if os.path.isfile(checkpoint_path):  # 检查文件是否存在
                logger.info(f"🔄 Loading checkpoint from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)    # 加载 checkpoint
                # 恢复状态
                start_epoch = checkpoint['epoch'] + 1
                best_mre = checkpoint['best_mre']
                coarse_net.load_state_dict(checkpoint['coarse_state_dict'])
                fine_LSTM.load_state_dict(checkpoint['fine_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"✅ Resumed training from Epoch {start_epoch}. Previous Best MRE: {best_mre:.4f}")
            else:
                logger.warning(f"⚠️ No checkpoint found at '{checkpoint_path}'. Starting from scratch.")

        # --- 训练循环 ---
        for epoch in range(start_epoch, config.epochs):     # 从start_epoch开始训练
            print() # 每轮开始在控制台打印一个空行隔开，log里面不用管
            train_coarse_Off = []
            train_fine_Off = []
            test_coarse_Off = []
            test_fine_Off = []
            
            for phase in ['train', 'val']:
                datas = dataloaders[phase]  # 直接获取 DataLoader
                pbar = tqdm(total=len(datas), desc=f'{phase} Epoch {epoch}') # 手动管理 tqdm，保持原始风格

                if phase == 'train':
                    coarse_net.train(True) # 开启训练模式
                    fine_LSTM.train(True)
                else:
                    if epoch % test_epoch != 0: continue
                    coarse_net.train(False)    # 开启测试模式
                    fine_LSTM.train(False)

                lent = len(datas)
                running_loss = 0

                # 遍历数据
                for data in datas:
                    inputs = data['DICOM'].cuda(config.use_gpu) # (B, C, D, H, W)
                    inputs_origin_list = data['DICOM_origin']
                    
                    # ⚠️ 注意: DataLoader 如果 batch_size > 1，DICOM_origin 可能会被 stack 成 Tensor
                    # 旧代码逻辑似乎期望它是一个 list of tensors。
                    # 如果 legacy_dataset.py 里没有改 collate_fn，这里 inputs_origin_list 可能是一个 5D Tensor (B, C, D, H, W)
                    # 下面这行旧代码暗示 inputs_origin_list 是一个 list 或者 Tensor
                    # inputs_origin = [item.squeeze(0) for item in inputs_origin_list] 
                    # 为了安全，我们假设 legacy_dataset 的输出是符合这里预期的
                    # 如果 inputs_origin_list 是 tensor (B, 1, 512, 512, 512)
                    if isinstance(inputs_origin_list, torch.Tensor):        # 如果是tensor
                        inputs_origin = [inputs_origin_list[b] for b in range(inputs_origin_list.shape[0])]
                    else:       # 如果是list
                        inputs_origin = [item.squeeze(0) for item in inputs_origin_list]

                    labels = data['landmarks'].cuda(config.use_gpu) 
               
                    size = data['size'][0]
                    px_z, px_y, px_x = size[0].item(), size[1].item(), size[2].item()
                    
                    # 构造像素尺寸张量
                    size_tensor_pixel = torch.tensor([px_x, px_y, px_z]).float().cuda(config.use_gpu).unsqueeze(0)   # Label 顺序是 [X, Y, Z],缩放因子也必须对应 [Width, Height, Depth]
                    size_tensor_inv = 1.0 / size_tensor_pixel.float()

                    # 物理尺寸张量 (用于计算 MRE 毫米误差)  逻辑: 像素数 * Spacing = 毫米数
                    sp_z, sp_y, sp_x = config.spacing # 从配置读取
                    physical_scale = torch.tensor([
                        px_x * sp_x, # Width (mm)
                        px_y * sp_y, # Height (mm)
                        px_z * sp_z  # Depth (mm)
                    ]).float().cuda(config.use_gpu).unsqueeze(0)
                    
                    optimizer.zero_grad()

                    # 显存控制：验证阶段不构建计算图 防止验证集吃掉显存
                    with torch.set_grad_enabled(phase == 'train'):
                        # 前向传播 (Forward)
                        coarse_heatmap, coarse_feature = coarse_net(inputs)

                        # 第一道关卡：检查网络输出是否正常 
                        has_nan = any(torch.isnan(h).any() for h in coarse_heatmap)     # 检查 list 中任何一个 tensor 是否有 NaN
                        if has_nan:     # 如果 CoarseNet 输出里就有 NaN，说明网络内部炸了 此时必须跳过，不能把 NaN 传给 MyUtils，否则会爆内存
                            logger.warning(f"⚠️ [Warning] NaN detected in CoarseNet output at Epoch {epoch}. Skipping this batch.")
                            optimizer.zero_grad() # 清空梯度
                            continue # 🔥 直接跳过！不跑 FineNet，不反向传播
                        
                        # 获取粗定位坐标
                        coarse_landmarks = MyUtils.get_coordinates_from_coarse_heatmaps(coarse_heatmap, global_coordinate)      # 这里有没有.unsqueeze(0)
                        # 强制限制坐标在 0-1 ，防止预测跑出边界
                        coarse_landmarks = torch.clamp(coarse_landmarks, 0.0, 1.0)
                        
                        # Fine Stage
                        fine_landmarks_all = fine_LSTM(coarse_landmarks, labels, inputs_origin, coarse_feature, phase, size_tensor_inv)

                        # 计算 Loss (Original Logic)
                        mask_loss = (labels[:, :, 0] >= 0).float().unsqueeze(2)
                        # 取最后一次迭代的结果来计算损失
                        fine_pred_last = fine_landmarks_all[-1].unsqueeze(0)

                        # Fine Loss: 使用 SmoothL1Loss
                        smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')    # 注意: reduction='none' 配合 mask 手动求和
                        loss_fine_raw = smooth_l1_loss(fine_pred_last, labels)
                        loss_fine = (loss_fine_raw * mask_loss).sum() / (mask_loss.sum() + 1e-6)
                        # Coarse Loss: 保持原样 (Heatmap Loss)
                        loss_coarse = criterion_coarse(coarse_heatmap, global_coordinate, labels, phase)
                        # Total Loss: 加权求和 (FineNet 权重翻倍)
                        loss = 2.0 * loss_fine + 1.0 * loss_coarse      # 目的: 告诉模型 "Coarse 差不多就行，但 Fine 必须准"

                        # 反向传播
                        if phase == 'train' and config.stage == 'train':
                            # 第二道关卡：检查 Loss 是否正常
                            if torch.isnan(loss):
                                logger.warning(f"⚠️ [Warning] Loss is NaN at Epoch {epoch}. Skipping gradient update.")
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
                        fine_landmarks_final = fine_landmarks_all[-1].unsqueeze(0)    # 只取最后一次迭代的结果用于评估
                        coarse_off = MyUtils.get_coarse_errors(coarse_landmarks, labels, physical_scale) # 计算误差，直接传入物理尺寸（mm）
                        fine_off = MyUtils.get_fine_errors(fine_landmarks_final, labels, physical_scale)

                        # 处理缺失值：填充 NaN
                        mask_valid = (labels[:, :, 0] >= 0) # [1, N]
                        coarse_off[~mask_valid] = float('nan')
                        fine_off[~mask_valid] = float('nan')
                        
                        if phase == "train":
                            train_fine_Off.append(fine_off.detach().cpu())
                            train_coarse_Off.append(coarse_off.detach().cpu())
                        else:
                            test_fine_Off.append(fine_off.detach().cpu())
                            test_coarse_Off.append(coarse_off.detach().cpu())
                    
                    running_loss += loss.item()
                    pbar.update(1)

                # End of Epoch
                epoch_loss = running_loss / lent
                pbar.close()
                
                if epoch % 1 == 0:
                    logger.info('{} epoch: {} Loss: {:.4f}'.format(phase, epoch, epoch_loss))
                
                if phase == 'train':
                    writer.add_scalar('Loss/Train', epoch_loss, epoch)
                elif phase == 'val':
                    writer.add_scalar('Loss/Val', epoch_loss, epoch)

            # -------------------------------------------------------------------
            # 5. TensorBoard 记录与结果保存
            # -------------------------------------------------------------------
            if epoch % test_epoch == 0:
                current_test_mre = float('inf')
                # --- 内部函数：计算并记录指标 (避免代码重复) ---
                def process_stats(tensor_list, prefix):
                    if len(tensor_list) == 0: return float('inf'), float('inf'), None
                    # 1. 拼接 [Total_N, 7] (包含 NaN)
                    all_tensor = torch.cat(tensor_list, dim=0)
                    # 2. MyUtils 计算细节 (SDR, 每列均值)
                    SDR, _, _ = MyUtils.analysis_result(config.landmarkNum, all_tensor)
                    # 3. 计算全局指标 (Micro-Average, 忽略 NaN)
                    global_mre = torch.nanmean(all_tensor).item()
                    # 计算全局 SD (兼容旧版 PyTorch 的手动 nanstd)
                    global_sd = torch.std(all_tensor[~torch.isnan(all_tensor)]).item()
                    # 4. 记录 TensorBoard
                    writer.add_scalar(f'{prefix}_MRE', global_mre, epoch)
                    writer.add_scalar(f'{prefix}_SD', global_sd, epoch)
                    # 记录 SDR (取所有关键点平均)
                    sdr_mean = torch.mean(SDR, dim=0) * 100
                    writer.add_scalar(f'{prefix}_SDR/2.0mm', sdr_mean[1], epoch) # 假设阈值索引1对应2.0mm
                    writer.add_scalar(f'{prefix}_SDR/4.0mm', sdr_mean[3], epoch)
                    writer.add_scalar(f'{prefix}_SDR/6.0mm', sdr_mean[5], epoch)
                    writer.add_scalar(f'{prefix}_SDR/8.0mm', sdr_mean[7], epoch)
                    return global_mre, global_sd, sdr_mean
                # --- 内部函数：输出指标结果 (避免代码重复) ---
                def print_detailed_results(title, c_mre, c_sd, f_mre, f_sd, sdr_vec):
                    logger.info(f"   [{title} Results]")
                    logger.info(f"   Fine   -> MRE: {f_mre:.4f} mm | SD: {f_sd:.4f} mm")
                    logger.info(f"   Coarse -> MRE: {c_mre:.4f} mm | SD: {c_sd:.4f} mm")
                    logger.info(f"   SDR (Thresholds):")
                    logger.info(f"     2.0mm:{sdr_vec[1]:.2f}%")
                    logger.info(f"     4.0mm:{sdr_vec[3]:.2f}%")
                    logger.info(f"     6.0mm:{sdr_vec[5]:.2f}%")
                    logger.info(f"     8.0mm:{sdr_vec[7]:.2f}%")
                    logger.info(f"   Full SDR Vector: {sdr_vec.tolist()}")
                    logger.info("")
                    
                # 处理 Train
                c_mre_train, c_sd_train, _ = process_stats(train_coarse_Off, 'Train/Coarse')
                f_mre_train, f_sd_train, sdr_train = process_stats(train_fine_Off, 'Train/Fine')

                # 处理 Val/Test 结果
                c_mre_test, c_sd_test, _ = process_stats(test_coarse_Off, 'Test/Coarse')
                f_mre_test, f_sd_test, sdr_test = process_stats(test_fine_Off, 'Test/Fine')
                
                # 记录对比曲线
                writer.add_scalars('Comparison/MRE_Train', {'Coarse': c_mre_train, 'Fine': f_mre_train}, epoch)
                writer.add_scalars('Comparison/MRE_Test', {'Coarse': c_mre_test, 'Fine': f_mre_test}, epoch)

                current_test_mre = f_mre_test # 以 Fine MRE 为准

                # 保存最佳模型
                if current_test_mre < best_mre:
                    logger.info(f"🔥 New Best! MRE: {best_mre:.4f} -> {current_test_mre:.4f} (Epoch {epoch})")
                    best_mre = current_test_mre
                    torch.save(coarse_net.state_dict(), os.path.join(save_dir, 'best_coarse.pth'))
                    torch.save(fine_LSTM.state_dict(), os.path.join(save_dir, 'best_fine_LSTM.pth'))
                
                if (epoch + 1) % 1 == 0:       # 每10个epoch打印输出一下评价指标
                    logger.info("")
                    print_detailed_results("TRAIN", c_mre_train, c_sd_train, f_mre_train, f_sd_train, sdr_train)
                    print_detailed_results("TEST ", c_mre_test, c_sd_test, f_mre_test, f_sd_test, sdr_test)

                # --- 🔥 保存断点 (包含优化器状态和Epoch) ---
                checkpoint_state = {
                    'epoch': epoch,
                    'coarse_state_dict': coarse_net.state_dict(),
                    'fine_state_dict': fine_LSTM.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), # 保存优化器状态
                    'best_mre': best_mre
                }
                torch.save(checkpoint_state, os.path.join(save_dir, 'latest_checkpoint.pth'), _use_new_zipfile_serialization=False)
                logger.info(f"💾 Checkpoint saved: epoch {epoch}")

            logger.info("")     # 打印空行，为了终端显示美观，日志里面会有个空行
            torch.cuda.empty_cache()
        
        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        writer.close()
    
    def test(self):
        """
        对应旧代码中的 test_model 函数
        """
        # 本地变量映射
        config = self.config
        coarse_net = self.coarse_net
        fine_LSTM = self.fine_LSTM
        dataloader = self.dataloaders['val'] # 假设 test 使用 val_loader

        # ==========================================
        # 👇 下面开始完全复刻 test_model 逻辑
        # ==========================================
        since = time.time()

        coarse_net.eval()
        fine_LSTM.eval()
        
        # 1. 准备全局坐标网格 (与 TrainNet 保持一致)
        gl, gh, gw = config.image_scale
        global_coordinate = torch.ones(gl, gh, gw, 3).float()
        for i in range(gl): global_coordinate[i, :, :, 0] *= i
        for i in range(gh): global_coordinate[:, i, :, 1] *= i
        for i in range(gw): global_coordinate[:, :, i, 2] *= i
        global_coordinate = global_coordinate.cuda(config.use_gpu) * torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)]).cuda(config.use_gpu)

        # 容器
        coarse_Off = []
        fine_Off = []

        # --- 1. 准备保存路径 ---
        # 注意：这里用 testName，如果没有定义在 config 里，可能需要检查一下
        # 为了兼容，如果 config 没有 testName，就用 saveName + '_test'
        test_name = getattr(config, 'testName', config.saveName + '_test')
        save_dir = os.path.join('runs', test_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # --- 2. 初始化 Logger ---   日志文件将保存在 runs/你的实验名/test.log
        log_path = os.path.join(save_dir, 'test.log')
        # 调用我们在 MyUtils 里写的函数
        logger = MyUtils.get_logger(log_path) 
        logger.info(f"🚀 Start Testing: {test_name}")
        logger.info(f"📁 Logs will be saved to: {save_dir}")
        landmark_names = ["AICV", "ANS", "BEP", "PNS", "TEE", "TEP", "TUV"]
        header = f"{'Sample ID':<25} | {'Coarse MRE':<12} | {'Fine MRE':<12} | {'Scale (mm)':<25}"
        for name in landmark_names: header += f" || {name[:4]+'_C':<10} | {name[:4]+'_F':<10}"     # 保持列宽紧凑，_C代表Coarse, _F代表Fine
        logger.info(header)
        logger.info("-" * len(header))

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs = data['DICOM'].cuda(config.use_gpu)
                labels = data['landmarks'].cuda(config.use_gpu)
                inputs_origin_list = data['DICOM_origin']
                
                # 兼容 tensor/list 输入
                if isinstance(inputs_origin_list, torch.Tensor):
                    inputs_origin = [inputs_origin_list[b] for b in range(inputs_origin_list.shape[0])]
                else:
                    inputs_origin = [item.squeeze(0) for item in inputs_origin_list]
                
                # --- 构建尺寸张量 ---
                # 1. 获取 Batch 中第一个样本的尺寸 (假设 Batch=1)
                size_data = data['size'][0] 
                px_z, px_y, px_x = size_data[0].item(), size_data[1].item(), size_data[2].item()
                
                # 2. 像素尺寸张量 (用于 FineNet 输入/归一化) - 顺序 [W, H, D] 对应 [x, y, z]
                size_tensor_pixel = torch.tensor([px_x, px_y, px_z]).float().cuda(config.use_gpu).unsqueeze(0)
                size_tensor_inv = 1.0 / size_tensor_pixel

                # 3. 物理尺寸张量 (用于计算 MRE 毫米误差)
                sp_z, sp_y, sp_x = config.spacing
                physical_scale = torch.tensor([
                    px_x * sp_x, # Width (mm)
                    px_y * sp_y, # Height (mm)
                    px_z * sp_z  # Depth (mm)
                ]).float().cuda(config.use_gpu).unsqueeze(0)

                # --- 推理 ---
                # 1. Coarse Stage
                coarse_heatmap, coarse_feature = coarse_net(inputs)
                
                # 2. Get Coarse Coordinates
                # coarse_landmarks = MyUtils.get_coordinates_from_coarse_heatmaps(coarse_heatmap, global_coordinate).unsqueeze(0)
                coarse_landmarks = MyUtils.get_coordinates_from_coarse_heatmaps(coarse_heatmap, global_coordinate)
                coarse_landmarks = torch.clamp(coarse_landmarks, 0.0, 1.0)
                
                # 3. Fine Stage
                fine_landmarks = fine_LSTM(coarse_landmarks, labels, inputs_origin, coarse_feature, 'test', size_tensor_inv)    # 这里得到的三次迭代的三个结果
                fine_landmarks = fine_landmarks[-1].unsqueeze(0)    # 只取最后一次迭代的结果

                # --- 计算误差 (mm) ---     这里的err形状是 (B, N)，包含了无效点的巨大误差
                c_err = MyUtils.get_coarse_errors(coarse_landmarks, labels, physical_scale)
                f_err = MyUtils.get_fine_errors(fine_landmarks, labels, physical_scale)

                # 使用 Mask 过滤无效点，保留有效点，无效位置设置为 nan
                mask = (labels[:, :, 0] >= 0) # 形状 (B, N)
                c_err[~mask] = float('nan')
                f_err[~mask] = float('nan')
                
                # 记录数据
                coarse_Off.append(c_err.cpu())
                fine_Off.append(f_err.cpu())
                
                # 打印单个样本信息 (可选)
                sample_name = data['imageName'][0] if 'imageName' in data else "Unknown"
                c_mre_sample = torch.nanmean(c_err).item()      # torch.nanmean可以直接处理 tensor中的 nan
                f_mre_sample = torch.nanmean(f_err).item()
                scale_str = f"[{physical_scale[0,0]:.1f}, {physical_scale[0,1]:.1f}, {physical_scale[0,2]:.1f}]"
                row_str = f"{sample_name[:25]:<25} | {c_mre_sample:<12.4f} | {f_mre_sample:<12.4f} | {scale_str:<25}"     # 基础信息字符串    
                # 横向追加每个点的误差
                c_vec = c_err[0].cpu().tolist()
                f_vec = f_err[0].cpu().tolist()
                
                for k in range(config.landmarkNum):
                    val_c = c_vec[k]
                    val_f = f_vec[k]
                    # 处理 NaN 显示
                    str_c = f"{val_c:.2f}" if val_c == val_c else "nan" # 保留2位小数节省空间
                    str_f = f"{val_f:.2f}" if val_f == val_f else "nan"
                    row_str += f" || {str_c:<10} | {str_f:<10}"
                
                logger.info(row_str)
                if (i + 1) % 10 == 0: logger.info("")

        # --- 最终统计分析 ---
        # logger.info("="*50)
        logger.info("="*len(header)) # 保持宽度一致
        logger.info("📊 Final Test Results")
        logger.info("="*len(header)) # 保持宽度一致
        # logger.info("="*50)
        
        if len(fine_Off) > 0:
            coarse_Off = torch.cat(coarse_Off, dim=0)
            fine_Off = torch.cat(fine_Off, dim=0)

            # 使用 MyUtils.analysis_result 进行统计，返回的是一个列表，每一列表示一个关键点的MRE、SD和SDR
            c_SDR, c_SD, c_MRE_list = MyUtils.analysis_result(config.landmarkNum, coarse_Off)
            f_SDR, f_SD, f_MRE_list = MyUtils.analysis_result(config.landmarkNum, fine_Off)

            # 计算全局均值 (对所有关键点的 MRE 再求一次平均) , analysis_result返回的是一个列表，代表7个关键点各自的平均误差
            c_final_mre = torch.nanmean(coarse_Off).item()  # 最科学的计算方法是将所有有效误差放在一起再求一次平均
            c_final_sd = torch.std(coarse_Off[~torch.isnan(coarse_Off)]).item() # SD最合理的计算不是标准差的平均值，而是计算全局标准差，直接对tensor求标准差，而不是对SD列表求平均值
            f_final_mre = torch.nanmean(fine_Off).item()
            f_final_sd = torch.std(fine_Off[~torch.isnan(fine_Off)]).item()
            
            logger.info(f"✅ Global Statistics:")
            logger.info(f"   Coarse Stage -> MRE: {c_final_mre:.4f} ± {c_final_sd:.4f} mm")
            logger.info(f"   Fine   Stage -> MRE: {f_final_mre:.4f} ± {f_final_sd:.4f} mm")
            logger.info("-" * 71)

            # --- 详细的关键点统计表 (Per-Landmark Table) ---
            logger.info(f"✅ Per-Landmark Statistics (Average over {len(fine_Off)} samples):")
            sum_header = f"{'ID':<4} | {'Name':<6} || {'Coarse MRE':<12} | {'Coarse SD':<10} || {'Fine MRE':<12} | {'Fine SD':<10}"
            logger.info(sum_header)
            logger.info("-" * len(sum_header))

            # 遍历每个关键点输出
            for k in range(config.landmarkNum):
                name = landmark_names[k]
                # 处理 Tensor 或 float
                cm = c_MRE_list[k] if isinstance(c_MRE_list, list) else c_MRE_list[k].item()
                cs = c_SD[k] if isinstance(c_SD, list) else c_SD[k].item()
                fm = f_MRE_list[k] if isinstance(f_MRE_list, list) else f_MRE_list[k].item()
                fs = f_SD[k] if isinstance(f_SD, list) else f_SD[k].item()
                logger.info(f"{k:<4} | {name:<6} || {cm:<12.4f} | {cs:<10.4f} || {fm:<12.4f} | {fs:<10.4f}")
            logger.info("-" * len(sum_header))
            
            # 打印详细 SDR
            logger.info(f"✅ Fine Stage SDR (Success Detection Rate):")
            # 取所有关键点 SDR 的平均值作为全局 SDR
            mean_sdr = torch.mean(f_SDR, dim=0) * 100
            thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # 对应 MyUtils 里的定义
            sdr_str_list = [f"{th}mm: {val:.2f}%" for th, val in zip(thresholds, mean_sdr)]
            for i in range(0, len(sdr_str_list), 4):
                logger.info("   " + " | ".join(sdr_str_list[i:i+4]))
            
        # logger.info("="*50)
        logger.info("="*len(header))

        time_elapsed = time.time() - since
        logger.info('test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))