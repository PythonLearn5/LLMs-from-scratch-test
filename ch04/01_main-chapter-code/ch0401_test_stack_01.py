import torch

a = torch.tensor([90, 80])
b = torch.tensor([85, 95])
c = torch.tensor([88, 92])

print(a.shape)
# torch.stack()（新增一个维度）
batch = torch.stack([a, b, c], dim=0)

print(batch)
print(batch.shape)
# dim=1 新增维度的位置不同、把这个新维度插入到 dim 指定的位置
result = torch.stack([a, b, c], dim=1)

print(result)
print(result.shape)
# torch.cat()（不增加维度）
result = torch.cat([a, b, c], dim=0)

print(result)
print(result.shape)