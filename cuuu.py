import torch

if __name__ == '__main__':
    # 测试CUDA
    print("Support CUDA?:", torch.cuda.is_available())
    x = torch.tensor([10.0])
    x = x.cuda()
    print(x)

    y = torch.randn(2, 3)
    y = y.cuda()
    print(y)

    z = x + y
    print(z)
