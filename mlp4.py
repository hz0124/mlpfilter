import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# 加载数据
data_x = np.loadtxt('data_x.txt', dtype=complex)
data_y = np.loadtxt('data_y.txt', dtype=complex)

data_x_real = data_x.real
data_y_real = data_y.real

# 超参数
window_size = 20  # 滑动窗口大小
input_size = window_size  # 输入大小 (复数被拆成实部和虚部)
hidden_size = 50  # 隐藏层大小
hidden_size2 = 100  # 隐藏层大小
output_size = 1  # 输出大小 (复数被拆成实部和虚部)
num_epochs = 10000  # 训练轮数
learning_rate = 0.001  # 学习率
batch_size = 64  # 批次大小

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

# 实例化模型并移动到设备上
model = MLP(input_size, hidden_size, hidden_size2, output_size).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 准备训练数据
def prepare_data(input_data, target_data, window_size):
    X, Y = [], []
    for i in range(len(input_data) - window_size):
        window_data = input_data[i:i + window_size]
        X.append(window_data)  # 将复数拆成实部和虚部
        Y.append(target_data[i + window_size])  # 输出也是复数
    return np.array(X), np.array(Y)

X, Y = prepare_data(data_x_real, data_y_real, window_size)

# 标准化数据
mean_X, std_X = np.mean(X, axis=0), np.std(X, axis=0)
mean_Y, std_Y = np.mean(Y), np.std(Y)

X = (X - mean_X) / std_X
Y = (Y - mean_Y) / std_Y

# 转换为Tensor并移动到设备上
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
best_loss = float('inf')
patience, trials = 100, 0

for epoch in range(num_epochs):
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    model.train()
    epoch_loss = 0
    for batch_X, batch_Y in dataloader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        trials = 0
    else:
        trials += 1

    if trials >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# 计算误差并保存
model.eval()
with torch.no_grad():
    outputs = model(X.to(device))
    outputs = outputs.cpu() * std_Y + mean_Y  # 反标准化并移动到CPU
    errors = (Y.cpu() * std_Y + mean_Y) - outputs  # 反标准化误差并移动到CPU

# 将误差保存到文件
errors = errors.numpy()
np.savetxt('error_prac3.txt', errors, fmt='%.6f')
print('Errors saved to error_prac.txt')
