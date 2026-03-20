# 注意力机制（Attention） 是现代大模型（GPT、BERT、Transformer）的核心机制
# 作用：当模型处理某个词时，决定应该关注句子中的哪些词

import torch

# 6个token，每个token用3维向量表示
# 实际大模型中通常是768维 / 1024维等
# 这里为了演示，用3维

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x1)
   [0.55, 0.87, 0.66], # journey  (x2)
   [0.57, 0.85, 0.64], # starts   (x3)
   [0.22, 0.58, 0.33], # with     (x4)
   [0.77, 0.25, 0.10], # one      (x5)
   [0.05, 0.80, 0.55]] # step     (x6)
)

# 选择第二个词作为 Query
# 在 self-attention 中，当前处理的词就是 query
query = inputs[1]  # journey

# 创建一个空tensor，用来存储注意力分数
attn_scores_2 = torch.empty(inputs.shape[0])

# 计算 query 和所有 token 的相似度
# 这里使用的是 dot product（点积）
for i, x_i in enumerate(inputs):
    # dot product = 相似度
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

# ---------------------------------------
# 方法2：手写 softmax
# ---------------------------------------

def softmax_naive(x):
    # softmax公式
    # e^xi / sum(e^x)
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
