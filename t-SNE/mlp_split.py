import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 定义降维和分类的 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super(MLP, self).__init__()

        # 第一部分：降维部分（编码器）
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),    # 从 10000 维到 512 维
            nn.ReLU(),                     # 激活函数
            nn.Dropout(0.9),                # Dropout 层，防止过拟合
            nn.Linear(512, intermediate_size)  # 从 512 维到 50 维（降维）
        )
        
        # 第二部分：分类部分（解码器）
        self.decoder = nn.Sequential(
            nn.Linear(intermediate_size, 128),  # 从 50 维到 128 维
            nn.ReLU(),                          # 激活函数
            nn.Dropout(0.5),                     # Dropout 层
            nn.Linear(128, output_size)          # 从 128 维到 20 类的输出
        )

    def forward(self, x):
        x = self.encoder(x)  # 编码部分：降维
        x = self.decoder(x)  # 解码部分：分类
        return x
    
    def get_encoded(self, x):
        return self.encoder(x)  


def evaluate(model, validation_loader, criterion):
    model.eval()  # 设置为评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(validation_loader)
    accuracy = 100 * correct / total
    return avg_val_loss, accuracy


def train(model, train_loader, validation_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = evaluate(model, validation_loader, criterion)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # 保存模型
        torch.save(model.state_dict(), f'model/mlp_model_{epoch+1}.pth')


def mlp_data():
    # 加载数据
    train_features = pickle.load(open('data/train_feature.pkl', 'rb'))
    train_labels = np.load('data/train_labels.npy')

    # 转换为 tensor 格式
    train_features = torch.tensor(train_features.toarray(), dtype=torch.float32)
    train_labels = torch.tensor(train_labels)

    # 数据划分：训练集和验证集
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features.numpy(), train_labels.numpy(), test_size=0.2, random_state=42
    )

    # 转换为 tensor 格式
    train_features = torch.tensor(train_features, dtype=torch.float32)
    val_features = torch.tensor(val_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_labels), batch_size=32, shuffle=False)

    # 定义模型、损失函数和优化器
    model = MLP(10000, 50, 20).cuda()
    criterion = nn.CrossEntropyLoss()  # 分类任务
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-10)

    # 训练模型
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)

if __name__ == '__main__':
    mlp_data()