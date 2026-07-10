import torch
import torch.nn as nn
# nn.Linear 做的是：用一个可学习的权重矩阵和偏置，对输入向量进行线性变换，把原始特征映射到新的特征空间
linear = nn.Linear(3, 2)
# 创建 Linear 时,PyTorch 会随机初始化它们、训练时会更新这些weight、bias
print(linear.weight)
print(linear.bias)
# 手动设置权重
linear.weight.data = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
])

# 手动设置 bias
linear.bias.data = torch.tensor([10., 20.])

x = torch.tensor([2., 3., 4.])

# 第一行权重： [1., 2., 3.]
# 第一个输出： 2×1 + 3×2 + 4×3 = 20
# 再加 bias 20+10 = 30

# 第二行权重： [4., 5., 6.]
# 第二个输出： 2×4 + 3×5 + 4×6 = 47
# 再加 bias 47+20 = 67
print(linear(x))