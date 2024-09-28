# 读取data_x，data_y中的数据，用data_x1中的数据拟合data_y1中的数据，训练一个多层感知机模型。

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# 加载数据
data_x = np.loadtxt('data_x1.txt')
data_y = np.loadtxt('data_y1.txt')


window_size = 50  # 滑动窗口大小
input_size = window_size  # 输入大小 (复数被拆成实部和虚部)
hidden_size = 256  # 隐藏层大小
hidden_size2 = 128  # 隐藏层大小
hidden_size3 = 64  # 隐藏层大小
hidden_size4 = 32  # 隐藏层大小
hidden_size5 = 16  # 隐藏层大小
output_size = 1  # 输出大小 (复数被拆成实部和虚部)
num_epochs = 100  # 训练轮数
learning_rate = 0.001  # 学习率

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_size5, output_size)

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
        out = self.relu5(out)
        out = self.fc6(out)
        return out

# 实例化模型并移动到设备上
model = MLP(input_size, hidden_size, hidden_size2, output_size).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# 准备训练数据
def prepare_data(input_data, target_data, window_size):
    X, Y = [], []
    for i in range(len(input_data) - window_size):
        window_data = input_data[i:i + window_size]
        X.append(window_data)  # 将复数拆成实部和虚部
        Y.append(target_data[i + window_size])  # 输出也是复数
    return np.array(X), np.array(Y)

X, Y = prepare_data(data_x, data_y, window_size)

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

# 创建数组，记录每次迭代的损失，用于绘图
losses = []

# 训练模型
loss = None  # Initialize loss variable
for epoch in range(num_epochs):
    model.train()
    for i, (x, y) in enumerate(loader):
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(loader)}], Loss: {loss.item():.4f}')

    # if (epoch + 1) % 100 == 0 and loss is not None:
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}')
    print (f'LR: {scheduler.get_last_lr()}')

    scheduler.step()


# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 读取模型
model = MLP(input_size, hidden_size, hidden_size2, output_size).to(device)
model.load_state_dict(torch.load('model.pth'))

# 计算误差并保存
model.eval()
with torch.no_grad():
    outputs = model(X)
    outputs = outputs.cpu() * std_Y + mean_Y  # 反标准化并移动到CPU
    errors = (Y.cpu() * std_Y + mean_Y) - outputs  # 反标准

# 绘制损失曲线
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('loss3.png')
print('Loss curve saved to loss3.png')

# 绘制预测结果
plt.figure()
plt.plot(data_y, label='True')
plt.plot(outputs, label='Predicted')
plt.legend()
plt.title('Prediction')
plt.grid(True)
plt.savefig('prediction2.png')
print('Prediction saved to prediction2.png')



