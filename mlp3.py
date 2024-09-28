import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# 加载数据
data_x = np.loadtxt('data_x.txt', dtype=complex)
data_y = np.loadtxt('data_y.txt', dtype=complex)

print("Data X sample:", data_x[:5])
print("Data Y sample:", data_y[:5])

# 超参数
window_size = 20  # 滑动窗口大小
input_size = window_size * 2  # 输入大小 (复数被拆成实部和虚部)
hidden_size = 50  # 隐藏层大小
output_size = 2  # 输出大小 (复数被拆成实部和虚部)
num_epochs = 3000  # 训练轮数
learning_rate = 0.001  # 学习率
batch_size = 64  # 批次大小
patience = 100  # 提前停止的耐心值

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

# 实例化模型并移动到设备上
model = MLP(input_size, hidden_size, output_size).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# 准备训练数据
def prepare_data(input_data, target_data, window_size):
    X, Y = [], []
    for i in range(len(input_data) - window_size):
        window_data = input_data[i:i + window_size]
        X.append(np.hstack((window_data.real, window_data.imag)))  # 将复数拆成实部和虚部
        Y.append([target_data[i + window_size].real, target_data[i + window_size].imag])  # 输出也是复数
    return np.array(X), np.array(Y)

X, Y = prepare_data(data_x, data_y, window_size)

print("Prepared X shape:", X.shape)
print("Prepared Y shape:", Y.shape)

# 标准化数据
mean_X, std_X = np.mean(X, axis=0), np.std(X, axis=0)
mean_Y, std_Y = np.mean(Y, axis=0), np.std(Y, axis=0)

X = (X - mean_X) / std_X
Y = (Y - mean_Y) / std_Y

print("Standardized X sample:", X[:5])
print("Standardized Y sample:", Y[:5])

# 转换为Tensor并移动到设备上
X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.float32).to(device)

# 创建数据加载器
dataset = torch.utils.data.TensorDataset(X, Y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
best_loss = float('inf')
trials = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()

    epoch_loss /= len(dataloader)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

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
    outputs = model(X)
    outputs = outputs.cpu() * std_Y + mean_Y  # 反标准化并移动到CPU
    errors = (Y.cpu() * std_Y + mean_Y) - outputs  # 反标准化误差并移动到CPU

# 将误差保存到文件
errors = errors.numpy()
np.savetxt('error_prac2.txt', errors, fmt='%.6f')
