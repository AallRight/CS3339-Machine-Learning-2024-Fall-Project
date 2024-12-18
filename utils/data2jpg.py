import pickle
import numpy as np
from PIL import Image
import os

# 创建保存图像的文件夹
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# 加载train_feature.pkl
with open('data/train_feature.pkl', 'rb') as f:
    train_features = pickle.load(f)

# 假设train_features是一个稀疏矩阵，转换为密集矩阵
train_features = train_features.toarray()  # 转换为密集矩阵

# 选择前10张图片
for i in range(min(10, len(train_features))):  # 只保存前10张图片
    # 将每个样本的10000维向量转换为100x100的图像
    img_data = train_features[i].reshape(100, 100)
    
    # 转换为灰度图像并保存为JPG格式
    img = Image.fromarray(img_data.astype(np.uint8))  # 转为8位灰度图像
    img = img.convert('L')  # 确保图像是灰度模式
    
    # 保存为JPEG文件
    img.save(os.path.join(output_dir, f"image_{i+1}.jpg"))

print("前10张图片已保存为JPG格式！")
