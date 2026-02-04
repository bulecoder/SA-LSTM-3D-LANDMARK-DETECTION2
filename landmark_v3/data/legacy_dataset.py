from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from skimage import io
import pandas as pd
import cv2
from utils import legacy_utils as MyUtils

class Rescale(object):
    """
    将 CSV 中的绝对坐标 (0~512) 归一化到 (0~1) 之间。
    """
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, sample):
        DICOM, DICOM_origin, landmarks, imageName = sample['DICOM'], sample['DICOM_origin'], sample['landmarks'], sample['imageName']
        
        # self.input_size 应该是 (512, 512, 512)
        d, h, w = self.input_size
        
        # 归一化坐标
        landmarks = landmarks / [d, h, w] 
        
        return {'DICOM': DICOM, 'DICOM_origin': DICOM_origin, 'landmarks': landmarks, 'imageName': imageName}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        DICOM, DICOM_origin, landmarks, imageName = sample['DICOM'], sample['DICOM_origin'], sample['landmarks'], sample['imageName']

        # 获取尺寸
        shape = np.array(DICOM_origin.shape) # (512, 512, 512)
        
        # --- 准备切图坐标 ---
        crop_landmarks_pixel = landmarks.copy() 
        
        # [A] 处理缺失值 (-1) -> 设为中心
        missing_mask = crop_landmarks_pixel[:, 0] < 0
        center = shape / 2.0
        crop_landmarks_pixel[missing_mask] = center
        
        # [B] 安全钳位 (Clamp)
        CROP_SIZE = 96
        SAFE_MARGIN = CROP_SIZE // 2
        
        min_limit = SAFE_MARGIN
        max_limit = shape - SAFE_MARGIN
        
        # 执行钳位
        crop_landmarks_pixel = np.clip(crop_landmarks_pixel, min_limit, max_limit)

        # --- 执行切 Patch ---
        # 必须转为 Tensor 才能传给 MyUtils
        DICOM_origin_tensor = torch.from_numpy(DICOM_origin).float().unsqueeze(0).unsqueeze(0)
        
        # 传入像素坐标 (float32)
        crop_coords_input = crop_landmarks_pixel.reshape(1, -1, 3).astype(np.float32)
        
        # 调用工具函数 (注意 useGPU=-1 表示 CPU)
        crop_list = MyUtils.getcropedInputs(crop_coords_input, DICOM_origin_tensor, CROP_SIZE, -1)

        # --- 返回 ---
        # 归一化坐标
        landmarks_normalized = landmarks / shape

        return {
            'DICOM': torch.from_numpy(DICOM).float().unsqueeze(0), 
            'DICOM_origin': crop_list, # 这是一个 list of tensors
            # 'DICOM_origin_vis': DICOM, # 为了节省内存，如果不用可视化可以注释掉
            'landmarks': torch.from_numpy(landmarks_normalized).float(), 
            'size': shape, 
            'imageName': imageName
        }


class LandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, landmarksNum=7):
        print(f"📖 Loading Dataset from: {csv_file}")
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.landmarkNum = landmarksNum
        print(f"   Found {len(self.landmarks_frame)} samples.")

    def __len__(self):
        return len(self.landmarks_frame)

    # 朴素归一化 (保留你的默认选择)
    def _minmax_normalize_cbct(self, image):
        image = image.astype(np.float32)
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 1e-5:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = image - min_val
        return image

    def __getitem__(self, idx):
        filename = self.landmarks_frame.iloc[idx, 0]
        
        # 🔍 打印进度 (每 10 个样本打印一次，防止刷屏)
        if idx % 10 == 0:
            print(f"   Loading sample [{idx}/{len(self)}]: {filename}")
        
        img_name_coarse = os.path.join(self.root_dir, "96_" + filename)
        img_name_fine = os.path.join(self.root_dir, filename)
        
        try:
            image_coarse = np.load(img_name_coarse)  
            image_fine = np.load(img_name_fine)  
            # 归一化
            image_coarse = self._minmax_normalize_cbct(image_coarse)
            image_fine = self._minmax_normalize_cbct(image_fine)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            # 返回 None 或者 raise，Trainer 里需要处理 None
            # 为了简单，直接 raise
            raise e

        landmarks = self.landmarks_frame.iloc[idx, 1:self.landmarkNum * 3 + 1].values.astype('float')
        landmarks = landmarks.reshape(-1, 3)

        sample = {
            'DICOM': image_coarse, 
            'DICOM_origin': image_fine, 
            'landmarks': landmarks, 
            'imageName': filename
        }

        if self.transform:
            sample = self.transform(sample)

        return sample