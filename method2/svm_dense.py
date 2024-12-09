import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# 1. 加载训练数据
with open('./data/train_feature.pkl', 'rb') as f:
    train_features = pickle.load(f)

train_features = train_features.toarray()  # 稀疏矩阵转为稠密矩阵

train_labels = np.load('./data/train_labels.npy')

# 2. 数据预处理：标准化（SVM对特征尺度敏感）
scaler = StandardScaler(with_mean=False)  # 稀疏数据需要设置 with_mean=False
train_features = scaler.fit_transform(train_features)

# 3. 使用 KFold 进行交叉验证，将数据分为 5 个批次
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5折交叉验证

fold = 1
accuracies = []

for train_idx, val_idx in kf.split(train_features):
    print(f'Fold {fold}...')

    # 训练集和验证集
    X_train, X_val = train_features[train_idx], train_features[val_idx]
    y_train, y_val = train_labels[train_idx], train_labels[val_idx]

    # 4. 使用支持向量机（SVM）进行训练
    svm_model = SVC(kernel="linear")  # 使用线性核SVM，适合高维稀疏数据
    svm_model.fit(X_train, y_train)

    # 创建SVM模型
    svm_model_select = SVC()
    print(svm_model.get_params())

    # 5. 在验证集上进行预测
    y_pred = svm_model.predict(X_val)

    # 6. 计算预测正确率
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)

    # 输出当前折的预测准确率
    print(f'Fold {fold} Accuracy: {accuracy:.4f}')

    # 7. 保存当前折的SVM模型

    model_save_path = f'model/method2/svm_model_fold_{timestamp}_{fold}.pkl'
    joblib.dump(svm_model, model_save_path)
    print(f'SVM model for fold {fold} saved to "{model_save_path}"')

    fold += 1

# 输出所有折的平均准确率
print(f'Average Accuracy across all folds: {np.mean(accuracies):.4f}')
print(f'类别:{svm_model.classes_}')
