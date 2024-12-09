import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd  # 导入 pandas 库
from datetime import datetime

# 1. 加载测试数据
with open('./data/test_feature.pkl', 'rb') as f:
    test_features = pickle.load(f)

test_features = test_features.toarray()  # 稀疏矩阵转为稠密矩阵

# 2. 加载已训练好的SVM模型
svm_model = joblib.load('model/method2/svm_model_2024-12-09_13-14-06.pkl')  # 从文件中加载模型

# # 3. 加载训练数据的Scaler并进行测试数据的标准化
# # 假设你之前训练模型时保存了Scaler
# scaler = joblib.load('scaler.pkl')  # 加载Scaler
# test_features = scaler.transform(test_features)
# 2. 数据预处理：标准化（SVM对特征尺度敏感）
scaler = StandardScaler(with_mean=False)  # 稀疏数据需要设置 with_mean=False
test_features = scaler.fit_transform(test_features)

# 4. 在测试集上进行预测
y_pred = svm_model.predict(test_features)

# 5. 保存预测结果到 CSV 文件
# 创建一个 DataFrame 包含ID和预测标签（label）
predictions_df = pd.DataFrame({
    'ID': np.arange(len(y_pred)),  # 使用索引作为ID
    'label': y_pred  # 预测标签
})

# 将 DataFrame 保存为 CSV 文件，文件名为 predictions.csv
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
predictions_df.to_csv(f'predictions/method2/predictions_{timestamp}.csv', index=False)
print(f'Predictions saved to "predictions_{timestamp}.csv"')
