import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data_x = np.loadtxt('data_x.txt', dtype=complex)
data_y = np.loadtxt('data_y.txt', dtype=complex)

# 提取一部分数据
start_index = 0  # 起始索引
end_index = 2000  # 结束索引

partial_data_x = data_x[start_index:end_index]
partial_data_y = data_y[start_index:end_index]

# 分别提取实部和虚部
real_x = partial_data_x.real
imag_x = partial_data_x.imag
real_y = partial_data_y.real
imag_y = partial_data_y.imag

# 作图
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(real_x, label='Real part of data_x')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Real part of data_x')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(imag_x, label='Imaginary part of data_x')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Imaginary part of data_x')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(real_y, label='Real part of data_y')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Real part of data_y')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(imag_y, label='Imaginary part of data_y')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Imaginary part of data_y')
plt.legend()

plt.tight_layout()
plt.show()
