import numpy as np

# 编写LMS算法,输入为复数
def lms(x, d, N=4, mu=0.1):
    nIters = min(len(x), len(d)) - N
    u = np.zeros(N, dtype=complex)  # 初始化为复数数组
    w = np.zeros(N, dtype=complex)  # 初始化为复数数组
    e = np.zeros(nIters, dtype=complex)  # 初始化为复数数组
    for n in range(nIters):
        u[1:] = u[:-1]
        u[0] = x[n]
        e_n = d[n] - np.dot(u, w)
        w = w + mu * e_n * np.conj(u)/(np.dot(u, np.conj(u))+1e-3)
        e[n] = e_n
        # 学习率递减
        #mu = mu * 0.9
    return e

# 加载数据
data_x = np.loadtxt('data_x.txt', dtype=complex)
data_y = np.loadtxt('data_y.txt', dtype=complex)

# 带入lms算法
en = lms(data_x, data_y, N=100, mu=0.001)
print(f"en[10001:10010]: {en[10001:10010]}")

en_real = en.real

# 计算平均误差
en_mean = np.mean(np.abs(en_real))
print(f"en: {en_mean}")

# 绘制误差信号
import matplotlib.pyplot as plt
plt.plot(np.abs(en))
plt.xlabel('Iteration')

# 显示图像
plt.show()
