# 用vottera算法拟合信号

import numpy as np

def svf(x, d, M=128, L=10, mu1=0.2, mu2=0.2):
  nIters = min(len(x),len(d)) - M
  L2=int(L*(L+1)/2)
  u = np.zeros(M)
  u2 = np.zeros((M,L2))
  w = np.zeros(M)
  h2 = np.zeros(L2)
  e = np.zeros(nIters)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    u2_n = np.outer(u[:L],u[:L])
    u2_n = u2_n[np.triu_indices_from(u2_n)]
    u2[1:] = u2[:-1]
    u2[0] = u2_n
    x2 = np.dot(u2,h2)
    g = u + x2
    y = np.dot(w, g.T)
    e_n = d[n] - y
    w = w + mu1*e_n*g/(np.dot(g,g)+1e-3)
    grad_2 = np.dot(u2.T,w)
    h2 = h2 + mu2*e_n*grad_2/(np.dot(grad_2,grad_2)+1e-3)
    e[n] = e_n
  return e

data_x = np.loadtxt('data_x.txt', dtype=complex)
data_y = np.loadtxt('data_y.txt', dtype=complex)

# 拆分实部虚部
data_x_real = data_x.real
data_x_imag = data_x.imag
data_y_real = data_y.real
data_y_imag = data_y.imag

# 带入svf算法
en_real = svf(data_x_real, data_y_real, M=128, L=10, mu1=0.2, mu2=0.2)
en_imag = svf(data_x_imag, data_y_imag, M=128, L=10, mu1=0.2, mu2=0.2)
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