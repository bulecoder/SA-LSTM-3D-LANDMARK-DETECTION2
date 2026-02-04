from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from skimage import io
import pandas as pd
import MyUtils
import zipfile
import cv2

class ZipDataset(Dataset):
    def __init__(self, root_path, cache_into_memory=False):
        if cache_into_memory:
            f = open(root_path, 'rb')
            self.zip_content = f.read()
            f.close()
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')
        else:
            self.zip_file = zipfile.ZipFile(root_path, 'r')
        self.name_list = list(filter(lambda x: x[-4:] == '.jpg', self.zip_file.namelist()))
        self.to_tensor = ToTensor()

    def __getitem__(self, key):
        buf = self.zip_file.read(name=self.name_list[key])
        img = self.to_tensor(cv2.imdecode(np.fromstring(buf, dtype=np.uint8), cv2.IMREAD_COLOR))
        return img

    def __len__(self):
        return len(self.name_list)


'''
if __name__ == '__main__':
    dataset = ZipDataset('COCO.zip', cache_into_memory=False)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    for batch_idx, sample in enumerate(dataloader):
        print(batch_idx, sample.size())
'''


class Rescale(object):
    """
    将 CSV 中的绝对坐标 (0~512) 归一化到 (0~1) 之间。
    注意：input_size 必须是 Fine Stage 的尺寸 (512, 512, 512)。
    """
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, sample):
        DICOM, DICOM_origin, landmarks, imageName = sample['DICOM'], sample['DICOM_origin'], sample['landmarks'], sample['imageName']
        
        # self.input_size 应该是 (512, 512, 512)
        d, h, w = self.input_size
        
        # 归一化坐标：x' = x / width
        # 注意：为了保持 -1 (缺失值) 仍然是负数，直接除以尺寸即可
        # 假设 landmarks 顺序是 x, y, z (对应 W, H, D)
        # 根据我们 prepare_data 的逻辑，我们存的是 voxel 坐标 (d0, d1, d2) 也就是 (D, H, W) 或者是 (x, y, z)?
        # MONAI 这里的坐标系通常是 RAS。我们之前的脚本存的是 numpy index (D, H, W)。
        # 所以我们需要按 (D, H, W) 的尺寸来归一化。
        # 既然我们设定是立方体 (512, 512, 512)，那么除以哪个都一样。
        
        landmarks = landmarks / [d, h, w] 
        
        return {'DICOM': DICOM, 'DICOM_origin': DICOM_origin, 'landmarks': landmarks, 'imageName': imageName}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # 1. 解包数据
        DICOM_origin = sample['DICOM_origin']
        landmarks = sample['landmarks']
        imageName = sample['imageName']
        
        # 获取 Dataset 传过来的 size
        # 如果 sample 里没有 size，就用 shape 计算
        shape = sample.get('size', np.array(DICOM_origin.shape))

        # 2. 转换为 Tensor (只转类型，不切图，不归一化)
        # image: (D, H, W) -> (1, D, H, W)
        img_tensor = torch.from_numpy(DICOM_origin).float().unsqueeze(0)
        
        # landmarks: 保持物理坐标 (N, 3)
        lm_tensor = torch.from_numpy(landmarks).float()

        # 3. 构建返回字典
        new_sample = {
            'DICOM_origin': img_tensor, 
            'landmarks': lm_tensor,
            'size': shape, 
            'imageName': imageName
        }
        
        # 4. 安全检查：如果未来某个时候 sample 里有了 'DICOM'，也顺便转一下
        # 这样写不会报 KeyError
        if 'DICOM' in sample:
            new_sample['DICOM'] = torch.from_numpy(sample['DICOM']).float().unsqueeze(0)

        return new_sample


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

    # HU阈值截断归一化
    def _clip_normalize_cbct(self, image):
        """
        专为 CBCT 设计的归一化函数  策略:Windowing [-1000, 1000] -> Normalize [0, 1]
        """
        image = image.astype(np.float32)    # 1. 转换为 float32 (节省显存，且 PyTorch 需要)
        MIN_HU = -1000.0
        MAX_HU = 1000.0
        image = np.clip(image, MIN_HU, MAX_HU)   # 2. 物理截断 (Windowing)   去掉金属伪影极高亮的影响，同时保留空气和骨骼的对比度
        image = (image - MIN_HU) / (MAX_HU - MIN_HU)    # 3. 归一化到 0~1
        return image

    # 朴素归一化
    def _minmax_normalize_cbct(self, image):
        """
        策略 B: 朴素方法。完全依赖当前图片的最大最小值。
        """
        image = image.astype(np.float32)
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 1e-5:
            image -= min_val            # 原地减
            image /= (max_val - min_val) # 原地除  防止除以 0 (虽然概率很小，但必须有)
        else:
            image -= min_val # 或者全变 0
        return image

    def __getitem__(self, idx):
        filename = self.landmarks_frame.iloc[idx, 0]
        
        # 🔍 打印进度 (每 10 个样本打印一次，防止刷屏)
        if idx % 10 == 0:
            print(f"   Loading sample [{idx}/{len(self)}]: {filename}")
        
        # img_name_coarse = os.path.join(self.root_dir, "96_" + filename)
        img_name_fine = os.path.join(self.root_dir, filename)
        
        try:
            # image_coarse = np.load(img_name_coarse)  
            image_fine = np.load(img_name_fine)  
            image_fine = self._minmax_normalize_cbct(image_fine)
            # 归一化
            # image_coarse = self._minmax_normalize_cbct(image_coarse)
            landmarks = self.landmarks_frame.iloc[idx, 1:self.landmarkNum * 3 + 1].values.astype('float')
            landmarks = landmarks.reshape(-1, 3)
            shape = np.array(image_fine.shape)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            raise e

        sample = {
            # 'DICOM': image_coarse, 
            'DICOM_origin': image_fine, 
            'landmarks': landmarks, 
            'imageName': filename,
            'size': shape
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
