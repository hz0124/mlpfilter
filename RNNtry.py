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

window_size = 50  # 滑动窗口大小
input_size = 1  # LSTM 每次输入一个时间步
hidden_size = 128  # 隐藏层大小
output_size = 1  # 输出大小
num_layers = 2  # LSTM 层数
num_epochs = 3  # 训练轮数
learning_rate = 0.001  # 学习率

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # 隐藏状态初始化
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # 细胞状态初始化

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 实例化模型并移动到设备上
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 准备训练数据
def prepare_data(input_data, target_data, window_size):
    X, Y = [], []
    for i in range(len(input_data) - window_size):
        window_data = input_data[i:i + window_size].reshape(-1, 1)
        X.append(window_data)
        Y.append(target_data[i + window_size])
    return np.array(X), np.array(Y)

X, Y = prepare_data(data_x_real, data_y_real, window_size)

# 标准化数据
mean_X, std_X = np.mean(X, axis=0), np.std(X, axis=0)
mean_Y, std_Y = np.mean(Y), np.std(Y)

X = (X - mean_X) / std_X
Y = (Y - mean_Y) / std_Y

# 转换为张量
X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(device)

# 创建数据加载器
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 训练模型
loss = None  # Initialize loss variable
for epoch in range(num_epochs):
    model.train()
    for i, (x, y) in enumerate(loader):
        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 保存模型
torch.save(model.state_dict(), 'model_lstm.pth')

# 预测
model.eval()
with torch.no_grad():
    X_test = X[:1, :window_size, :]  # 初始形状为 (1, window_size, input_size)
    Y_pred = []
    for i in range(len(X) - window_size):
        Y_test = model(X_test)  # 输出形状为 (1, 1)
        Y_pred.append(Y_test.item())

        # 确保 Y_test 的形状为 (1, 1, input_size) 以便拼接
        Y_test = Y_test.view(1, 1, 1)

        # 更新 X_test，保持形状为 (1, window_size, input_size)
        X_test = torch.cat((X_test[:, 1:, :], Y_test), dim=1)

# 还原预测值到原始尺度
Y_pred = np.array(Y_pred) * std_Y + mean_Y
Y_true = data_y_real[window_size:]

min_length = min(len(Y_pred), len(Y_true))
Y_pred = Y_pred[:min_length]
Y_true = Y_true[:min_length]

# 计算误差
error = np.mean(np.abs(Y_pred - Y_true))
print(f'Mean Absolute Error: {error:.6f}')
print(f"The 10000th error : {np.abs(Y_pred[10000] - Y_true[10000])}")




