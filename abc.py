import torch

# 分配一些GPU内存
x = torch.randn(10000, 10000).cuda()

# 查询已经分配的GPU内存大小
memory_allocated = torch.cuda.memory_allocated()

# 打印结果
print(memory_allocated)
