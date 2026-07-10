import torch
import torch.nn as nn

# # nn.Linear 做的是：用一个可学习的权重矩阵和偏置，对输入向量进行线性变换，把原始特征映射到新的特征空间
# 输入维度：3、输出维度：2
layer = nn.Linear(3, 2)

x = torch.tensor([[1.0, 2.0, 3.0]])
y = layer(x)

print(y)