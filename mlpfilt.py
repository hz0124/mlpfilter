import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# 加载数据
data_x = np.loadtxt('data_x.txt', dtype=complex)
data_y = np.loadtxt('data_y.txt', dtype=complex)

# 超参数
window_size = 20  # 滑动窗口大小
input_size = window_size * 2  # 输入大小 (复数被拆成实部和虚部)
hidden_size = 50  # 隐藏层大小
output_size = 2  # 输出大小 (复数被拆成实部和虚部)
num_epochs = 10000  # 训练轮数
learning_rate = 0.001  # 学习率

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out

# 实例化模型并移动到设备上
model = MLP(input_size, hidden_size, output_size).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 准备训练数据
def prepare_data(input_data, target_data, window_size):
    X, Y = [], []
    for i in range(len(input_data) - window_size):
        window_data = input_data[i:i + window_size]
        X.append(np.hstack((window_data.real, window_data.imag)))  # 将复数拆成实部和虚部
        Y.append([target_data[i + window_size].real, target_data[i + window_size].imag])  # 输出也是复数
    return np.array(X), np.array(Y)

X, Y = prepare_data(data_x, data_y, window_size)

# 转换为Tensor并移动到设备上
X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.float32).to(device)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 计算误差并保存
model.eval()
with torch.no_grad():
    outputs = model(X)
    errors = Y - outputs

# 将误差移动到CPU并保存
errors = errors.cpu().numpy()
np.savetxt('error_prac1.txt', errors, fmt='%.6f')
print('Errors saved to error_prac1.txt')
