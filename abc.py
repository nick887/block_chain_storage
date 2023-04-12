import numpy as np
import torch

# 生成NumPy数组
x_np = np.random.rand(1000, 1000)

# 将NumPy数组转换为PyTorch张量，并将其转移到GPU上
x = torch.from_numpy(x_np).to('cuda')

# 在GPU上进行计算
y = torch.sin(x)

# 将结果从GPU上转移到CPU上，并转换为NumPy数组
y_np = y.to('cpu').detach().numpy()

# 输出结果
print(y_np)
