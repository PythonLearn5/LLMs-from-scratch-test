import torch
import torch.nn as nn

# 输入维度：3、输出维度：2
layer = nn.Linear(3, 2)

x = torch.tensor([[1.0, 2.0, 3.0]])
y = layer(x)

print(y)