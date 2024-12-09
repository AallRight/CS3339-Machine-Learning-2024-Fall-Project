import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import joblib
from datetime import datetime
import pandas as pd

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 1. 加载训练数据
with open('./data/train_feature_dense.pkl', 'rb') as f:
    train_features = pickle.load(f)
test_features = pickle.load(open('./data/test_feature.pkl', 'rb'))
test_features = test_features.toarray()

# train_features = train_features.toarray()  # 稀疏矩阵转为稠密矩阵

train_labels = np.load('./data/train_labels.npy')

# 2. 数据预处理：标准化（SVM对特征尺度敏感）
# scaler = StandardScaler(with_mean=False)  # 稀疏数据需要设置 with_mean=False
# scaler = StandardScaler()
# train_features = scaler.fit_transform(train_features)

print(train_features.shape)

# 打乱数据
train_features, train_labels = shuffle(train_features, train_labels, random_state=42)

# 3. 将数据划分为训练集和测试集，1/7作为测试集
X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.01, random_state=42)

print(X_train.shape, X_test.shape)

# 4. 使用支持向量机（SVM）进行训练
svm_model = SVC(kernel='linear', decision_function_shape='ovr')  # 使用线性核SVM，适合高维稀疏数据
print(svm_model.get_params())
svm_model.fit(X_train, y_train)

# 5. 在测试集上进行预测
y_pred = svm_model.predict(X_test)
test_pred = svm_model.predict(test_features)

# 6. 计算预测正确率
accuracy = accuracy_score(y_test, y_pred)

# 输出预测正确率
print(f'Prediction Accuracy: {accuracy:.4f}')

# 7. 保存SVM模型
joblib.dump(svm_model, f'model/method2/svm_model_{timestamp}.pkl')
print(f'SVM model saved to "svm_model_{timestamp}.pkl"')

predictions_df = pd.DataFrame({
    'ID': np.arange(len(test_pred)),  # 使用索引作为ID
    'label': test_pred  # 预测标签
})

predictions_df.to_csv(f'predictions/method2/predictions_{timestamp}.csv', index=False)
print(f'Predictions saved to "predictions_{timestamp}.csv"')