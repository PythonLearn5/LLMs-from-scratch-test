import torch.nn as nn
import torch

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
        # do not exceed `context_length` before reaching this forward method.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

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

# inputs 的形状(shape)为：
# (6, 3)
# 表示：
# 有 6 个 token
# 每个 token 是 3 维向量
# 即：
# [
#   [0.43,0.15,0.89],   # token1
#   [0.55,0.87,0.66],   # token2
#   ...
# ]
# torch.stack()
# 功能：
# 将多个 Tensor 沿着新的维度拼接。
# 这里：
# (inputs, inputs)
# 表示把 inputs 复制两份。
#
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3

torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)