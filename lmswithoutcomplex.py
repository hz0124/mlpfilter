import numpy as np

def lms(x, d, N = 20, mu = 0.1):
  nIters = min(len(x),len(d)) - N
  u = np.zeros(N)
  w = np.zeros(N)
  e = np.zeros(nIters)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    e_n = d[n] - np.dot(u, w)
    w = w + mu * e_n * u / (np.dot(u, u) + 1e-3)
    e[n] = e_n
    # 学习率递减
    #mu = mu * 0.9
  return e

data_x = np.loadtxt('data_x.txt', dtype=complex)
data_y = np.loadtxt('data_y.txt', dtype=complex)

# 拆分实部虚部
data_x_real = data_x.real
data_x_imag = data_x.imag
data_y_real = data_y.real
data_y_imag = data_y.imag

# 带入lms算法
en_real = lms(data_x_real, data_y_real, N=20, mu=0.001)
en_imag = lms(data_x_imag, data_y_imag, N=20, mu=0.001)
print(en_real[10001:10010])
print(en_imag[10001:10010])

# 计算平均误差
en = np.mean(np.abs(en_real))
print(f"en: {en}")

# 绘制误差信号
import matplotlib.pyplot as plt
plt.plot(np.abs(en_real))
plt.xlabel('Iteration')

# 显示图像
plt.show()