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

# 将数据拆分为实部和虚部
data_x_real = data_x.real
data_x_imag = data_x.imag
data_y_real = data_y.real
data_y_imag = data_y.imag

window_size = 50  # 滑动窗口大小
input_size = window_size  # 输入大小 (实部或虚部)
hidden_size = 256  # 第一层隐藏层大小
hidden_size2 = 128  # 第二层隐藏层大小
hidden_size3 = 64   # 第三层隐藏层大小
hidden_size4 = 32   # 第四层隐藏层大小
hidden_size5 = 16   # 第五层隐藏层大小
output_size = 1     # 输出大小 (实部或虚部)
num_epochs = 3      # 训练轮数
learning_rate = 0.001  # 学习率

# 准备数据函数
def prepare_data(input_real, input_imag, target_real, target_imag, window_size):
    X_real, X_imag, Y_real, Y_imag = [], [], [], []
    for i in range(len(input_real) - window_size):
        window_real = input_real[i:i + window_size]
        window_imag = input_imag[i:i + window_size]
        X_real.append(window_real)
        X_imag.append(window_imag)
        Y_real.append(target_real[i + window_size])
        Y_imag.append(target_imag[i + window_size])
    return np.array(X_real), np.array(X_imag), np.array(Y_real), np.array(Y_imag)

# 准备数据
X_real, X_imag, Y_real, Y_imag = prepare_data(data_x_real, data_x_imag, data_y_real, data_y_imag, window_size)

# 标准化数据
mean_X_real, std_X_real = np.mean(X_real, axis=0), np.std(X_real, axis=0)
mean_X_imag, std_X_imag = np.mean(X_imag, axis=0), np.std(X_imag, axis=0)
mean_Y_real, std_Y_real = np.mean(Y_real), np.std(Y_real)
mean_Y_imag, std_Y_imag = np.mean(Y_imag), np.std(Y_imag)

X_real = (X_real - mean_X_real) / std_X_real
X_imag = (X_imag - mean_X_imag) / std_X_imag
Y_real = (Y_real - mean_Y_real) / std_Y_real
Y_imag = (Y_imag - mean_Y_imag) / std_Y_imag

# 转换为张量
X_real = torch.tensor(X_real, dtype=torch.float32).to(device)
X_imag = torch.tensor(X_imag, dtype=torch.float32).to(device)
Y_real = torch.tensor(Y_real, dtype=torch.float32).view(-1, 1).to(device)
Y_imag = torch.tensor(Y_imag, dtype=torch.float32).view(-1, 1).to(device)

# 创建数据加载器
dataset = TensorDataset(X_real, X_imag, Y_real, Y_imag)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 定义复数多层感知机模型
class ComplexMLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(ComplexMLP, self).__init__()
        # 实部网络
        self.fc1_real = nn.Linear(input_size, hidden_size)
        self.fc2_real = nn.Linear(hidden_size, hidden_size2)
        self.fc3_real = nn.Linear(hidden_size2, hidden_size3)
        self.fc4_real = nn.Linear(hidden_size3, hidden_size4)
        self.fc5_real = nn.Linear(hidden_size4, hidden_size5)
        self.fc6_real = nn.Linear(hidden_size5, output_size)

        # 虚部网络
        self.fc1_imag = nn.Linear(input_size, hidden_size)
        self.fc2_imag = nn.Linear(hidden_size, hidden_size2)
        self.fc3_imag = nn.Linear(hidden_size2, hidden_size3)
        self.fc4_imag = nn.Linear(hidden_size3, hidden_size4)
        self.fc5_imag = nn.Linear(hidden_size4, hidden_size5)
        self.fc6_imag = nn.Linear(hidden_size5, output_size)

        self.relu = nn.ReLU()

    def forward(self, x_real, x_imag):
        # 第一层
        real_out1 = self.fc1_real(x_real) - self.fc1_imag(x_imag)
        imag_out1 = self.fc1_real(x_imag) + self.fc1_imag(x_real)
        real_out1 = self.relu(real_out1)
        imag_out1 = self.relu(imag_out1)

        # 第二层
        real_out2 = self.fc2_real(real_out1) - self.fc2_imag(imag_out1)
        imag_out2 = self.fc2_real(imag_out1) + self.fc2_imag(real_out1)
        real_out2 = self.relu(real_out2)
        imag_out2 = self.relu(imag_out2)

        # 第三层
        real_out3 = self.fc3_real(real_out2) - self.fc3_imag(imag_out2)
        imag_out3 = self.fc3_real(imag_out2) + self.fc3_imag(real_out2)
        real_out3 = self.relu(real_out3)
        imag_out3 = self.relu(imag_out3)

        # 第四层
        real_out4 = self.fc4_real(real_out3) - self.fc4_imag(imag_out3)
        imag_out4 = self.fc4_real(imag_out3) + self.fc4_imag(real_out3)
        real_out4 = self.relu(real_out4)
        imag_out4 = self.relu(imag_out4)

        # 第五层
        real_out5 = self.fc5_real(real_out4) - self.fc5_imag(imag_out4)
        imag_out5 = self.fc5_real(imag_out4) + self.fc5_imag(real_out4)
        real_out5 = self.relu(real_out5)
        imag_out5 = self.relu(imag_out5)

        # 输出层
        real_out = self.fc6_real(real_out5) - self.fc6_imag(imag_out5)
        imag_out = self.fc6_real(imag_out5) + self.fc6_imag(real_out5)

        return real_out, imag_out

# 实例化模型并移动到设备上
model = ComplexMLP(input_size, hidden_size, hidden_size2, output_size).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# 训练模型
losses = []
loss = None
for epoch in range(num_epochs):
    model.train()
    for x_real, x_imag, y_real, y_imag in loader:
        # 前向传播
        real_outputs, imag_outputs = model(x_real, x_imag)
        loss_real = criterion(real_outputs, y_real)
        loss_imag = criterion(imag_outputs, y_imag)
        loss = loss_real + loss_imag

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}')
    print(f'LR: {scheduler.get_last_lr()}')
    scheduler.step()

# 保存模型
torch.save(model.state_dict(), 'model_comp.pth')

# 读取模型
model = ComplexMLP(input_size, hidden_size, hidden_size2, output_size).to(device)
model.load_state_dict(torch.load('model_comp.pth'))

# 计算误差并保存
model.eval()
with torch.no_grad():
    real_outputs, imag_outputs = model(X_real, X_imag)
    real_outputs = real_outputs.cpu() * std_Y_real + mean_Y_real  # 反标准化
    imag_outputs = imag_outputs.cpu() * std_Y_imag + mean_Y_imag  # 反标准化
    real_errors = (Y_real.cpu() * std_Y_real + mean_Y_real) - real_outputs
    imag_errors = (Y_imag.cpu() * std_Y_imag + mean_Y_imag) - imag_outputs

# 绘制损失曲线,对数坐标
import matplotlib.pyplot as plt
plt.plot(losses)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('loss7.png')
print('Loss curve saved to loss7.png')

# 显示图像
plt.show()

# 保存误差
np.savetxt('real_errors.txt', real_errors.numpy(), fmt='%.6f')
np.savetxt('imag_errors.txt', imag_errors.numpy(), fmt='%.6f')
print('Errors saved to real_errors.txt and imag_errors.txt')
