import argparse

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # --- 模型训练参数 ---
        self.parser.add_argument("--batchSize", type=int, default=1)
        self.parser.add_argument("--landmarkNum", type=int, default=7)
        # 注意：argparse 默认不支持 tuple 类型，通常用 nargs='+' 或者在代码里写死
        # 旧代码可能是直接把 tuple 默认值传进去了，或者你运行的时候没改这个参数
        # 为了兼容，我们在这里设为默认值，如果需要修改，建议用 nargs=3
        self.parser.add_argument("--image_scale", default=(96, 96, 96), type=tuple) 
        self.parser.add_argument("--origin_image_size", default=(512, 512, 512), type=tuple)
        self.parser.add_argument("--crop_size", default=(96, 96, 96), type=tuple)
        self.parser.add_argument("--use_gpu", type=int, default=0)
        self.parser.add_argument("--iteration", type=int, default=3)
        self.parser.add_argument("--epochs", type=int, default=50)
        self.parser.add_argument("--data_enhanceNum", type=int, default=1)
        self.parser.add_argument('--lr', type=float, default=0.0001)
        self.parser.add_argument("--spacing", default=(0.5, 0.5, 0.5), type=tuple)
        self.parser.add_argument("--stage", type=str, default="train")
        self.parser.add_argument("--resume", type=str, default=None)
        
        # --- 输入数据参数 ---
        self.parser.add_argument('--dataRoot', type=str, default="F:/CBCT/SA-LSTM-3D-Landmark-Detection2/processed_data/")
        self.parser.add_argument("--traincsv", type=str, default='train.csv') # 旧代码是 train1.csv? 请确认
        self.parser.add_argument("--testcsv", type=str, default='test.csv')   # 旧代码是 test1.csv? 请确认
        
        # --- 输出保存参数 ---
        self.parser.add_argument("--saveName", type=str, default='Refactor_Check_v3') # 改个名区分
        self.parser.add_argument("--testName", type=str, default="SmoothL1Loss_AdamW")

    def _parse_tuple(self, s):
        # 简单的字符串转 tuple 逻辑，或者直接返回默认值
        if isinstance(s, tuple): return s
        try:
            return tuple(map(float, s.strip('()').split(',')))
        except:
            return s

    def parse(self):
        return self.parser.parse_args()

# 使用示例: cfg = Config().parse()