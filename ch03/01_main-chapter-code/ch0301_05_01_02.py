import torch.nn as nn
import torch

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

# ========================================
# 输入数据
# ========================================
#
# 这里模拟一句话:
#
# Your journey starts with one step
#
# 每个词已经经过 embedding
#
# 每个词用3维向量表示
#
inputs = torch.tensor(
[
 [0.43, 0.15, 0.89],   # Your     x1
 [0.55, 0.87, 0.66],   # journey  x2
 [0.57, 0.85, 0.64],   # starts   x3
 [0.22, 0.58, 0.33],   # with     x4
 [0.77, 0.25, 0.10],   # one      x5
 [0.05, 0.80, 0.55]    # step     x6
]
)

# 输入维度
#
# 每个token:
# [0.43,0.15,0.89]
#
# 所以:
#
# d_in = 3
#
d_in = inputs.shape[1]

# 注意力内部计算维度
#
# Q/K/V都会转换成2维
#
d_out = 2

# 固定随机数
# 保证 Linear 初始化结果一致
torch.manual_seed(789)

# 创建自注意力对象
sa_v2 = SelfAttention_v2(
    d_in,
    d_out
)

# 输出每个token融合上下文后的向量
print(sa_v2(inputs))

# ===================================================
# 手动查看 Attention 权重
# ===================================================
# 计算 Q K
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)

# Query 和 Key 做矩阵乘法
#
# 得到:
#
# 6 x 6 矩阵
#
# 行:
# 当前token
#
# 列:
# 被关注token
#
attn_scores = queries @ keys.T

# 转换成概率
attn_weights = torch.softmax(
    attn_scores / keys.shape[-1] ** 0.5,
    dim=-1
)
print(attn_weights)

# ===================================================
# 因果注意力(Causal Attention)
# ===================================================
# GPT 类模型不能看到未来内容
# 例如:
# 输入:
# 我 喜欢 吃 苹果
# 计算"喜欢"时:
# 可以看:
# 我
# 喜欢
# 不能看:
# 吃
# 苹果
# 所以需要mask

context_length = attn_scores.shape[0]

# ---------------------------------------------
# 创建因果 Mask（Causal Mask）
# ---------------------------------------------

# torch.ones(context_length, context_length)
# 假设有 6 个 token，则先生成：
# 1 1 1 1 1 1
# 1 1 1 1 1 1
# 1 1 1 1 1 1
# 1 1 1 1 1 1
# 1 1 1 1 1 1
# 1 1 1 1 1 1
# torch.triu(..., diagonal=1)
# triu = Upper Triangle（保留上三角）
# diagonal=1 表示：
# 从主对角线上方开始保留
# 得到：
# 0 1 1 1 1 1
# 0 0 1 1 1 1
# 0 0 0 1 1 1
# 0 0 0 0 1 1
# 0 0 0 0 0 1
# 0 0 0 0 0 0
# 其中：
#   1 表示"未来 token"，需要屏蔽
#   0 表示允许关注
#
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)

# ---------------------------------------------
# 使用 masked_fill 屏蔽未来 token
# ---------------------------------------------
# mask.bool()
# 将数字：
# 0 -> False
# 1 -> True
# 变成布尔矩阵：
# False True True True True True
# False False True True True True
# False False False True True True
# masked_fill(condition, value)
# 表示： 如果 condition == True 就把当前位置替换成 value
# 为什么填充成 -inf？
# 因为后面要进入 softmax：
# softmax(x)=e^x / Σe^x
# 而： e^(-∞)=0
# 所以未来 token 的权重最终一定会变成 0
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
# ---------------------------------------------
# 计算 Attention 权重
# ---------------------------------------------
# 先除以 sqrt(d_k)
# d_k = Key 向量维度
# 为什么？
# 防止点积过大，使 softmax 输出过于极端， 导致梯度消失。 然后进入 softmax：
# 例如：
# score: [2  1 -inf]
# softmax 内部实际上计算：
# e² e¹ e^-∞ = 7.39 2.71 0
# 再归一化：
# 7.39 / 10.1 = 0.73
# 2.71 / 10.1 = 0.27
# 0    / 10.1 = 0
# 得到： [0.73  0.27  0]
# 可以看到： 未来 token 自动变成了 0
# 并且每一行仍然满足：
# 所有概率之和 = 1
# 所以这里不需要像上一种方法那样重新归一化。
#
attn_weights = torch.softmax(
    masked / keys.shape[-1] ** 0.5,
    dim=-1
)

print(attn_weights)
