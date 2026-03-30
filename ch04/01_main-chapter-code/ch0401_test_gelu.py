import torch.nn.functional as F
import torch

y = torch.tensor([
  [1, 2, 3, 0, 5, -6]
], dtype=torch.float32)
y = F.gelu(y)

#x = x.float()
# 或
#x = x.to(torch.float32)

print(y)

z = torch.tensor([1, 2, 3])
z = z.float()  # ✅ 转成 float
z = F.gelu(z)  # ✅ 用概率方式平滑地决定一个值该保留多少

print(z)