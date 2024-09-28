# 创建两组数据，其中一组是正弦函数，另一组是正弦函数加上随机噪声，并分别保存至两个文件中
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
num_samples = 1000
t = np.linspace(0, 2 * np.pi, num_samples)
sin_t = np.sin(t)
noise = np.random.normal(0, 0.1, num_samples)
sin_t_noisy = sin_t + noise

# 保存数据
np.savetxt('data_x1.txt', sin_t)
np.savetxt('data_y1.txt', sin_t_noisy)

# 绘制数据
plt.figure()
plt.plot(t, sin_t, label='sin(t)')
plt.plot(t, sin_t_noisy, label='sin(t) + noise')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig('data.png')
print('Data saved to data.png')
