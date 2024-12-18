import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 第一层卷积，输入通道1，输出通道32，卷积核3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 第二层卷积，输入通道32，输出通道64，卷积核3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # 池化层，2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        # Dropout层
        self.dropout = nn.Dropout(p=0.3)  # 50% dropout
        self.dropout2 = nn.Dropout(p=0.5)  # 50% dropout

        # 全连接层，输入维度要根据上面卷积池化后的输出维度来计算
        # 假设输入图片大小为100x100，经过两次2x2池化后，尺寸会变成25x25
        self.fc1 = nn.Linear(64 * 25 * 25, 512)  # 输入大小为64 * 25 * 25
        self.fc2 = nn.Linear(512, 20)  # 输出20个类别
        
    def forward(self, x):
        # 第一层卷积 -> 激活函数 -> 池化
        x = self.pool(torch.gelu(self.conv1(x)))
        # 第二层卷积 -> 激活函数 -> 池化
        x = self.pool(torch.gelu(self.conv2(x)))
        # 展开成一维向量
        x = x.view(-1, 64 * 25 * 25)
        # 全连接层 -> 激活函数 -> Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 添加dropout
        # 输出层
        x = self.fc2(x)
        return x

# 验证集评估函数
def evaluate(model, validation_loader, criterion):
    model.eval()  # 将模型设置为评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in validation_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(validation_loader)
    accuracy = 100 * correct / total
    return avg_val_loss, accuracy

# 模型训练函数
def train(model, train_loader, validation_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
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

            # 计算训练精度
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        # 计算训练损失和精度
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # 计算验证集损失和精度
        val_loss, val_accuracy = evaluate(model, validation_loader, criterion)

        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')


        # 保存模型
        torch.save(model.state_dict(), f'model/method1/cnn_model_{epoch+1}_{timestamp}.pth')

# 数据准备
if __name__ == '__main__':
    # 加载数据
    train_features = pickle.load(open('./data/train_feature.pkl', 'rb'))
    train_labels = np.load('./data/train_labels.npy')

    # 转换为 Tensor
    train_features = torch.tensor(train_features.toarray(), dtype=torch.float32)
    train_features = train_features.view(train_features.shape[0], 100, 100)
    train_features = train_features.unsqueeze(1)  # 增加一个维度，变成 (num_samples, 1, 100, 100)
    train_labels = torch.tensor(train_labels)

    # 拆分为训练集和验证集 (80% 训练, 20% 验证)
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features.numpy(), train_labels.numpy(), test_size=0.2, random_state=42)

    # 转换为 PyTorch 张量
    train_features = torch.tensor(train_features, dtype=torch.float32)
    val_features = torch.tensor(val_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    # 使用 TensorDataset 和 DataLoader 加载数据
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 创建模型、损失函数和优化器
    model = CNN().cuda()  # 将模型放到GPU上（如果有）
    criterion = nn.CrossEntropyLoss()  # 分类问题用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, train_loader, val_loader, criterion, optimizer, epochs=10)
