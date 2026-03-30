import torch.nn.functional as F
import torch

x = torch.tensor([
  [1, 2, 3, 0, 5, -6]
], dtype=torch.long)
# relu 函数把负数变成 0，正数保持不变
print(x)
x = F.relu(x)
print(x)
