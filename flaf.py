import numpy as np

def flaf(x, d, M=128, P=5, mu=0.2):
  nIters = min(len(x),len(d)) - M
  Q = P*2
  u = np.zeros(M)
  w = np.zeros((Q+1)*M)
  e = np.zeros(nIters)
  sk = np.zeros(P*M,dtype=np.int32)
  ck = np.zeros(P*M,dtype=np.int32)
  pk = np.tile(np.arange(P),M)
  for k in range(M):
    sk[k*P:(k+1)*P] = np.arange(1,Q,2) + k*(Q+1)
    ck[k*P:(k+1)*P] = np.arange(2,Q+1,2) + k*(Q+1)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    g = np.repeat(u,Q+1)
    g[sk] = np.sin(np.pi*pk*g[sk])
    g[ck] = np.cos(np.pi*pk*g[ck])
    y = np.dot(w, g.T)
    e_n = d[n] - y
    w = w + 2*mu*e_n*g/(np.dot(g,g)+1e-3)
    e[n] = e_n
  return e

data_x = np.loadtxt('data_x.txt', dtype=complex)
data_y = np.loadtxt('data_y.txt', dtype=complex)

# 拆分实部虚部
data_x_real = data_x.real
data_x_imag = data_x.imag
data_y_real = data_y.real
data_y_imag = data_y.imag

# 带入flaf算法
en_real = flaf(data_x_real, data_y_real, M=128, P=5, mu=0.2)
en_imag = flaf(data_x_imag, data_y_imag, M=128, P=5, mu=0.2)
print(en_real[10001:10010])
print(en_imag[10001:10010])

# 绘制误差信号
import matplotlib.pyplot as plt
plt.plot(np.abs(en_real))
plt.xlabel('Iteration')

# 显示图像
plt.show()
