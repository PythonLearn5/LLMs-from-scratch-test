import torch
import torch.nn.functional as F

# 默认这里是 int64
#
z = torch.tensor([1, 2, 3])

# GELU不能处理整数
# 所以需要转换成float
z = z.float()

# 等价于： z = z.to(torch.float32)
# 然后才能送入GELU

z = F.gelu(z)

print("\n整数转float后再经过GELU：")
print(z)