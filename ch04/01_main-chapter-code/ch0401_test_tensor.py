import torch

# 标量（0维） 就一个数
x = torch.tensor(5)
print(x)
print(x.shape)
# 向量（1维） 一排数据
x = torch.tensor([1, 2, 3])
print(x)
print(x.shape)
# 矩阵（2维）
x = torch.tensor([
    [1, 2],
    [3, 4]
])
print(x)
print(x.shape)
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
# x = torch.randn(2, 3, 4)
# 创建到第一个 GPU。需要你安装的 PyTorch 不支持 CUDA
# print(torch.__version__)         # PyTorch版本
# print(torch.version.cuda)        # 编译时使用的CUDA版本
# print(torch.cuda.is_available()) # 是否可以使用CUDA
# x = torch.randn(2, 3, 4, device="cuda:0")
print(x)
# 表示维度
print(x.shape)
# dtype（数据类型）
print(x.dtype)
# x.device 在哪里 CPU or GPU
print(x.device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(2, 3, 4, device=device)
print(x.device)