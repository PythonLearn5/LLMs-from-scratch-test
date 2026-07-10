import torch
import torch.nn as nn

# 随机丢弃 50%
dropout = nn.Dropout(p=0.5)

x = torch.tensor([1., 2., 3., 4.])

dropout.train()   # 训练模式

print(dropout(x))

torch.manual_seed(123)

example = torch.ones(3, 4) # 创建一个随机矩阵

print(dropout(example))