import torch
import torch.nn as nn


# =========================
# 自定义 LayerNorm 类
# =========================
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 避免除0
        # 可学习缩放参数 γ（初始化为1）
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 可学习平移参数 β（初始化为0）
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # x.shape = [batch_size, emb_dim]

        # 对最后一维求均值
        mean = x.mean(dim=-1, keepdim=True)

        # 对最后一维求方差
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 标准化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # 再做可学习的线性变换
        return self.scale * norm_x + self.shift


# =========================
# 设置随机种子，保证可复现
# =========================
torch.manual_seed(123)

# =========================
# 创建 2 个训练样本，每个样本 5 个特征
# =========================
batch_example = torch.randn(2, 5)  # shape = [2, 5]

# =========================
# 定义一个简单的线性层 + ReLU
# =========================
layer = nn.Sequential(
    nn.Linear(5, 6),  # 输入5维，输出6维
    nn.ReLU()  # 激活函数
)

# 前向传播
out = layer(batch_example)  # shape = [2, 6]
print(out)

# =========================
# 计算每个样本（最后一维）的均值和方差
# =========================
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)

# =========================
# 手动做标准化（未乘 γ 和 β）
# =========================
out_norm = (out - mean) / torch.sqrt(var)
print("Normalized layer outputs:\n", out_norm)

# 再次检查标准化后的均值和方差
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# 设置打印，不使用科学计数法
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# =========================
# 使用自定义 LayerNorm
# =========================
ln = LayerNorm(emb_dim=6)  # emb_dim=6 对应 out 的最后一维
out_ln = ln(out)

# 检查 LayerNorm 后的均值和方差
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)