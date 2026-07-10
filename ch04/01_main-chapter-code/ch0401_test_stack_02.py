import torch

# 构造 batch
batch = [
    torch.tensor([6109, 3626, 6100, 345]),
    torch.tensor([6109, 1110, 6622, 257])
]

# torch.stack()（新增一个维度）
batch = torch.stack(batch, dim=0)

print(batch)
print(batch.shape)
