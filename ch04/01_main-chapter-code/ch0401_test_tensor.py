import torch

# 标量（0维） 就一个数
x = torch.tensor(5)
print(x)
# 向量（1维） 一排数据
x = torch.tensor([1, 2, 3])
print(x)
# 矩阵（2维）
x = torch.tensor([
    [1, 2],
    [3, 4]
])
print(x)

# 高维 Tensor  生成一个 3维 Tensor,2 个 “3×4 的矩阵” 叠在一起
# [
#   [   # 第 1 个块
#     [a, b, c, d],
#     [e, f, g, h],
#     [i, j, k, l]
#   ],
#
#   [   # 第 2 个块
#     [m, n, o, p],
#     [q, r, s, t],
#     [u, v, w, x]
#   ]
# ]
x = torch.randn(2, 3, 4)
print(x)
# 表示维度
print(x.shape)
# dtype（数据类型）
print(x.dtype)
# x.device 在哪里 CPU or GPU
print(x.device)