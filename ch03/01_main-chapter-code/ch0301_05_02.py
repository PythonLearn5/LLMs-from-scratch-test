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

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # dropout rate of 50%
example = torch.ones(context_length, context_length) # create a matrix of ones

print(dropout(example))

torch.manual_seed(123)
print(dropout(attn_weights))