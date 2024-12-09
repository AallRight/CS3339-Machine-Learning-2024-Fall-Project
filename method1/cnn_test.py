import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle
from cnn import CNN
import datetime

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")   

# 假设你已经加载了模型，并准备好了 test_feature 数据
# 加载模型
model = CNN().cuda()  # 创建与训练时相同的模型结构，并放到GPU上
model.load_state_dict(torch.load('model/method1/cnn_model_6_2024-12-07_23-40-13.pth'))  # 加载训练好的模型权重
model.eval()  # 设置模型为评估模式

# 假设 test_feature 是形状为 (N, 10000) 的张量或 Numpy 数组
# 将 test_feature 转换为 PyTorch 张量
test_features = pickle.load(open('./data/test_feature.pkl', 'rb'))  # 加载测试数据
test_features = test_features.toarray()  # 如果是稀疏矩阵，转换为密集矩阵

# 转换为 Tensor，调整维度为 (num_samples, 1, 100, 100)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_features = test_features.view(test_features.shape[0], 100, 100)  # 将每个样本转换为 100x100
test_features = test_features.unsqueeze(1)  # 增加一个维度，变成 (num_samples, 1, 100, 100)

# 将数据加载到 DataLoader 中
batch_size = 64
test_dataset = TensorDataset(test_features)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 用模型对数据进行预测
predictions = []
with torch.no_grad():  # 禁用梯度计算，提高性能
    for inputs in test_loader:
        inputs = inputs[0].cuda()  # 获取输入数据并移到GPU
        outputs = model(inputs)  # 通过模型计算输出
        _, predicted = torch.max(outputs, 1)  # 获取预测结果 (类别)
        predictions.extend(predicted.cpu().numpy())  # 保存预测结果到CPU并转为NumPy数组

# 创建 DataFrame，将 ID 和预测标签合并
ids = np.arange(len(predictions))  # 假设样本的 ID 是从 0 到 len(predictions)-1
df = pd.DataFrame({
    'ID': ids,
    'label': predictions
})

# 保存为 CSV 文件
df.to_csv(f'predictions/method1/predictions_{timestamp}.csv', index=False)

print("Predictions saved to 'predictions.csv'")
