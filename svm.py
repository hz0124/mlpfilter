import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载数据
data_x = np.loadtxt('data_x.txt', dtype=complex)
data_y = np.loadtxt('data_y.txt', dtype=complex)

# 将数据拆分为实部和虚部
data_x_real = data_x.real
data_x_imag = data_x.imag
data_y_real = data_y.real
data_y_imag = data_y.imag

window_size = 50  # 滑动窗口大小

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

# 合并特征
X = np.column_stack((X_real, X_imag))

# 标准化数据
scaler_X = StandardScaler()
scaler_Y_real = StandardScaler()
scaler_Y_imag = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_real_scaled = scaler_Y_real.fit_transform(Y_real.reshape(-1, 1)).flatten()
Y_imag_scaled = scaler_Y_imag.fit_transform(Y_imag.reshape(-1, 1)).flatten()

# 创建支持向量回归模型
model_real = SVR(kernel='rbf')
model_imag = SVR(kernel='rbf')

# 训练模型
model_real.fit(X_scaled, Y_real_scaled)
model_imag.fit(X_scaled, Y_imag_scaled)

# 预测
Y_real_pred_scaled = model_real.predict(X_scaled)
Y_imag_pred_scaled = model_imag.predict(X_scaled)

# 反标准化
Y_real_pred = scaler_Y_real.inverse_transform(Y_real_pred_scaled.reshape(-1, 1)).flatten()
Y_imag_pred = scaler_Y_imag.inverse_transform(Y_imag_pred_scaled.reshape(-1, 1)).flatten()

# 计算误差
real_errors = Y_real - Y_real_pred
imag_errors = Y_imag - Y_imag_pred

# 绘制误差图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(real_errors, label='Real Part Errors')
plt.title('Real Part Errors')
plt.xlabel('Sample Index')
plt.ylabel('Error')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(imag_errors, label='Imaginary Part Errors')
plt.title('Imaginary Part Errors')
plt.xlabel('Sample Index')
plt.ylabel('Error')
plt.legend()

plt.tight_layout()
plt.show()

# 保存误差
np.savetxt('real_errors_svr.txt', real_errors, fmt='%.6f')
np.savetxt('imag_errors_svr.txt', imag_errors, fmt='%.6f')
