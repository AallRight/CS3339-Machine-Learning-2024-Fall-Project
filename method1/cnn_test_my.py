import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import pickle
import numpy as np
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class TrainDataset(Dataset):
    def __init__(self, feature_path, label_path):
        with open(feature_path, 'rb') as f:
            self.features = pickle.load(f)
        self.labels = np.load(label_path)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义卷积神经网络 (CNN) 结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 第一层卷积，输入通道1，输出通道32，卷积核3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 第二层卷积，输入通道32，输出通道64，卷积核3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # 池化层，2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 全连接层，输入维度要根据上面卷积池化后的输出维度来计算
        # 假设输入图片大小为100x100，经过两次2x2池化后，尺寸会变成25x25
        self.fc1 = nn.Linear(64 * 25 * 25, 512)  # 输入大小为64 * 25 * 25
        self.fc2 = nn.Linear(512, 20)  # 输出20个类别
        
    def forward(self, x):
        # 第一层卷积 -> 激活函数 -> 池化
        x = self.pool(torch.relu(self.conv1(x)))
        # 第二层卷积 -> 激活函数 -> 池化
        x = self.pool(torch.relu(self.conv2(x)))
        # 展开成一维向量
        x = x.view(-1, 64 * 25 * 25)
        # 全连接层 -> 激活函数
        x = torch.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x

# 模型训练函数
def train(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            # 送入 GPU 进行计算（如果有的话）
            inputs, labels = inputs.cuda(), labels.cuda()

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播并优化
            loss.backward()
            optimizer.step()

            # 计算精度
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
        torch.save(model.state_dict(), f'model/method1/cnn_model_{epoch+1}_{timestamp}.pth')

# 准备数据集
# 假设 train_features 为 (num_samples, 100, 100)，train_labels 为 (num_samples,)

if __name__ == '__main__':
    # 示例数据
    # train_features = np.random.rand(1000, 1, 100, 100).astype(np.float32)
    # train_labels = np.random.randint(0, 20, 1000)
    train_features = pickle.load(open('./data/train_feature.pkl', 'rb'))
    # print(train_features.shape)
    train_features = train_features.toarray()
    # print(train_features.shape)
    train_labels = np.load('./data/train_labels.npy')

    # 转换为 Tensor
    # train_features = torch.tensor(train_features)
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_features = train_features.view(train_features.shape[0], 100, 100) 
    train_features = train_features.unsqueeze(1)  # 增加一个维度，变成 (num_samples, 1, 100, 100)
    train_labels = torch.tensor(train_labels)

    # 使用 TensorDataset 和 DataLoader 加载数据
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 创建模型、损失函数和优化器
    model = CNN().cuda()  # 将模型放到GPU上（如果有）
    criterion = nn.CrossEntropyLoss()  # 分类问题用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, train_loader, criterion, optimizer, epochs=10)
