import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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
num_epochs = 3  # 训练轮数
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
mean_Y_real, std_Y_real = np.mean(Y_real), np.std(Y_real)
mean_Y_imag, std_Y_imag = np.mean(Y_imag), np.std(Y_imag)

Y_real = (Y_real - mean_Y_real) / std_Y_real
Y_imag = (Y_imag - mean_Y_imag) / std_Y_imag

# 转换为张量并调整形状
X = np.stack((X_real, X_imag), axis=1)  # 创建两个通道
X = torch.tensor(X, dtype=torch.float32).to(device).unsqueeze(1)  # 增加一个维度作为通道数
Y_real = torch.tensor(Y_real, dtype=torch.float32).view(-1, 1).to(device)
Y_imag = torch.tensor(Y_imag, dtype=torch.float32).view(-1, 1).to(device)

# 创建数据加载器
dataset = TensorDataset(X, Y_real, Y_imag)
loader = DataLoader(dataset, batch_size=64, shuffle=False)


# 定义卷积神经网络模型
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 2), padding=(1, 0))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 2), padding=(1, 0))
        self.fc1 = nn.Linear(32 * (window_size // 2) * 1, 256)
        self.fc_output_real = nn.Linear(256, 1)
        self.fc_output_imag = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)  # 池化层
        x = self.relu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)  # 池化层
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))

        real_out = self.fc_output_real(x)
        imag_out = self.fc_output_imag(x)

        return real_out, imag_out


# 实例化模型并移动到设备上
model = ComplexCNN().to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
loss = None
losses = []
for epoch in range(num_epochs):
    model.train()
    for i, (x, y_real, y_imag) in enumerate(loader):
        # 前向传播
        real_outputs, imag_outputs = model(x)
        loss_real = criterion(real_outputs, y_real)
        loss_imag = criterion(imag_outputs, y_imag)
        loss = loss_real + loss_imag

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if i % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(loader)}], Loss: {loss.item():.8f}')

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}')

# 保存模型
torch.save(model.state_dict(), 'model_comp_cnn.pth')

# 读取模型
model.load_state_dict(torch.load('model_comp_cnn.pth'))

# 计算误差并保存
model.eval()
with torch.no_grad():
    real_outputs, imag_outputs = model(X)
    real_outputs = real_outputs.cpu() * std_Y_real + mean_Y_real  # 反标准化
    imag_outputs = imag_outputs.cpu() * std_Y_imag + mean_Y_imag  # 反标准化
    real_errors = (Y_real.cpu() * std_Y_real + mean_Y_real) - real_outputs
    imag_errors = (Y_imag.cpu() * std_Y_imag + mean_Y_imag) - imag_outputs

# 绘制损失曲线
plt.plot(losses)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('loss_cnn.png')
plt.show()

# 保存误差
np.savetxt('real_errors_cnn.txt', real_errors.numpy(), fmt='%.6f')
np.savetxt('imag_errors_cnn.txt', imag_errors.numpy(), fmt='%.6f')
print('Errors saved to real_errors_cnn.txt and imag_errors_cnn.txt')