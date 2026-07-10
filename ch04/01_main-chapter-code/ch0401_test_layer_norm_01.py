import torch
import torch.nn as nn

# 一个样本，共6个特征
x = torch.tensor([[1., 2., 3., 4., 5., 6.]])

print("\n原始输入：")
print(x)

# 创建 LayerNorm
#
# normalized_shape=6
# 表示： 对最后一个维度(6个特征)做归一化.均值为0，标准差为1
ln = nn.LayerNorm(6)

# 执行 LayerNorm
y = ln(x)

print("\nLayerNorm 后：")
print(y)

# 查看均值
print("\n均值：")
print(y.mean(dim=-1))

# 查看标准差
print("\n标准差：")
print(y.std(dim=-1, unbiased=False))

# test
z = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
])
# 平均值告诉你中心在哪里
print(z.mean(dim=-1))
# 标准差告诉你数据偏离中心有多远。
print(z.std(dim=-1, unbiased=False))