import pandas as pd
import os
import numpy as np

# 配置
DATA_ROOT = "F:/CBCT/SA-LSTM-3D-Landmark-Detection2/processed_data/"
INPUT_CSV = os.path.join(DATA_ROOT, "processed_data.csv")
TRAIN_RATIO = 0.9

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到文件: {INPUT_CSV}")
        return

    print(f"📖 读取数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # 随机打乱
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 切分
    train_len = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:train_len]
    test_df = df.iloc[train_len:]
    
    # 保存
    train_path = os.path.join(DATA_ROOT, "train.csv")
    test_path = os.path.join(DATA_ROOT, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ 切分完成!")
    print(f"   Train: {len(train_df)} -> {train_path}")
    print(f"   Test:  {len(test_df)}  -> {test_path}")

if __name__ == "__main__":
    main()