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

window_size = 64  # 滑动窗口大小
input_size = window_size  # 输入大小 (实部或虚部)
hidden_size = 16  # 隐藏层大小
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


# 遍历X_real查看其中是否有绝对值小于0.00001的值
num = 0
for i in range(len(X_real)):
    if np.any(np.abs(X_real[i]) < 0.00001):
        num += 1
print(f"num: {num}")

print(f"shape of X_real: {X_real.shape}")

# 绘制X_real图像
plt.plot(X_real[10000:10010, 0])
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('X_real')
plt.grid(True)

# 显示图像
plt.show()

# 绘制X_real直方图
plt.hist(X_real[10000:11000, 0], bins=100)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of X_real')
plt.grid(True)

# 显示图像
plt.show()

# 转换为张量
X_real = torch.tensor(X_real, dtype=torch.float32).to(device)
X_imag = torch.tensor(X_imag, dtype=torch.float32).to(device)
Y_real = torch.tensor(Y_real, dtype=torch.float32).view(-1, 1).to(device)
Y_imag = torch.tensor(Y_imag, dtype=torch.float32).view(-1, 1).to(device)

# 创建数据加载器
dataset = TensorDataset(X_real, X_imag, Y_real, Y_imag)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 添加异常检测
torch.autograd.set_detect_anomaly(True)

# 定义复数残差块
class ComplexResidualBlock(nn.Module):
    def __init__(self, size):
        super(ComplexResidualBlock, self).__init__()
        self.fc_real = nn.Linear(size, size)
        self.fc_imag = nn.Linear(size, size)
        self.relu = nn.ReLU(inplace=False)  # 确保 inplace=False

    def forward(self, x_real, x_imag):
        real_out = self.fc_real(x_real) - self.fc_imag(x_imag)
        imag_out = self.fc_real(x_imag) + self.fc_imag(x_real)

        real_out = self.relu(real_out)
        imag_out = self.relu(imag_out)

        # 残差连接：不要直接修改输入，使用新的张量返回
        real_out = real_out + x_real
        imag_out = imag_out + x_imag

        return real_out, imag_out

# 定义复数多层感知机模型
class ComplexMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_residuals=3):
        super(ComplexMLP, self).__init__()
        # 第一层 (非残差)
        self.fc1_real = nn.Linear(input_size, hidden_size)
        self.fc1_imag = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=False)  # 确保 inplace=False

        # 9 层残差网络
        self.reidual_blocks = nn.ModuleList([ComplexResidualBlock(hidden_size) for _ in range(num_residuals)])

        #self.residual_block1 = ComplexResidualBlock(hidden_size)
        #self.residual_block2 = ComplexResidualBlock(hidden_size)
        #self.residual_block3 = ComplexResidualBlock(hidden_size)
        #self.residual_block4 = ComplexResidualBlock(hidden_size)
        #self.residual_block5 = ComplexResidualBlock(hidden_size)
        #self.residual_block6 = ComplexResidualBlock(hidden_size)
        #self.residual_block7 = ComplexResidualBlock(hidden_size)
        #self.residual_block8 = ComplexResidualBlock(hidden_size)
        #self.residual_block9 = ComplexResidualBlock(hidden_size)

        # 输出层
        self.fc_output_real = nn.Linear(hidden_size, output_size)
        self.fc_output_imag = nn.Linear(hidden_size, output_size)

    def forward(self, x_real, x_imag):
        # 第一层
        real_out1 = self.fc1_real(x_real) - self.fc1_imag(x_imag)
        imag_out1 = self.fc1_real(x_imag) + self.fc1_imag(x_real)
        real_out1 = self.relu(real_out1)
        imag_out1 = self.relu(imag_out1)

        # 残差块
        for block in self.reidual_blocks:
            real_out1, imag_out1 = block(real_out1, imag_out1)

        #real_out2, imag_out2 = self.residual_block1(real_out1, imag_out1)
        #real_out3, imag_out3 = self.residual_block2(real_out2, imag_out2)
        #real_out4, imag_out4 = self.residual_block3(real_out3, imag_out3)
        #real_out5, imag_out5 = self.residual_block4(real_out4, imag_out4)
        #real_out6, imag_out6 = self.residual_block5(real_out5, imag_out5)
        #real_out7, imag_out7 = self.residual_block6(real_out6, imag_out6)
        #real_out8, imag_out8 = self.residual_block7(real_out7, imag_out7)
        #real_out9, imag_out9 = self.residual_block8(real_out8, imag_out8)
        #real_out10, imag_out10 = self.residual_block9(real_out9, imag_out9)

        # 输出层
        real_out = self.fc_output_real(real_out1) - self.fc_output_imag(imag_out1)
        imag_out = self.fc_output_real(imag_out1) + self.fc_output_imag(real_out1)

        return real_out, imag_out

    """# 定义权重初始化函数，使用Kaiming初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)"""


# 实例化模型并移动到设备上
model = ComplexMLP(input_size, hidden_size, output_size).to(device)
#model.init_weights()

# 查看模型结构
print(model)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# 训练模型
losses = []
loss = None
weight_history = []
for epoch in range(num_epochs):
    model.train()
    for i , (x_real, x_imag, y_real, y_imag) in enumerate(loader):
        # 前向传播
        real_outputs, imag_outputs = model(x_real, x_imag)
        loss_real = criterion(real_outputs, y_real)
        loss_imag = criterion(imag_outputs, y_imag)
        loss = loss_real + loss_imag

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录权重
        if i % 100 == 0:
            weight_history.append(model.fc1_real.weight.data[0][:5].cpu().numpy())
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(loader)}], Loss: {loss.item():.8f}')

        losses.append(loss.item())
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(loader)}], Loss: {loss.item():.8f}')

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}')
    print(f'LR: {scheduler.get_last_lr()}')
    scheduler.step()

# 保存模型
torch.save(model.state_dict(), 'model_comp_resnet.pth')

# 读取模型
model.load_state_dict(torch.load('model_comp_resnet.pth'))

# 计算误差并保存
model.eval()
with torch.no_grad():
    real_outputs, imag_outputs = model(X_real, X_imag)
    real_outputs = real_outputs.cpu() * std_Y_real + mean_Y_real  # 反标准化
    imag_outputs = imag_outputs.cpu() * std_Y_imag + mean_Y_imag  # 反标准化
    real_errors = (Y_real.cpu() * std_Y_real + mean_Y_real) - real_outputs
    imag_errors = (Y_imag.cpu() * std_Y_imag + mean_Y_imag) - imag_outputs

# 绘制损失曲线,对数坐标
plt.plot(losses)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('loss_resnet1.png')
print('Loss curve saved to loss_resnet1.png')
plt.show()

# 可视化权重变化
weight_history = np.array(weight_history)
plt.figure(figsize=(10, 5))
for i in range(weight_history.shape[1]):
    plt.plot(weight_history[:, i], label=f'Weight {i+1}')
plt.xlabel('Iteration')
plt.ylabel('Weight Value')
plt.legend()
plt.savefig('weights_resnet.png')
print('Weights curve saved to weights_resnet.png')
plt.show()

# 打印第10000到第10010个real_errors
print(real_errors[10000:10010])

# 保存误差
np.savetxt('real_errors_resnet.txt', real_errors.numpy(), fmt='%.6f')
np.savetxt('imag_errors_resnet.txt', imag_errors.numpy(), fmt='%.6f')