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

# 创建一个空tensor，用来存储注意力分数 6 * 3  inputs.shape[0] = 6  inputs.shape[1] = 3
attn_scores_2 = torch.empty(inputs.shape[0])

# 计算 query 和所有 token 的相似度
# 这里使用的是 dot product（点积）
# x1 = [0.43, 0.15, 0.89]
# x2 = [0.55, 0.87, 0.66]
# x1 · x2 = a1*b1 + a2*b2 + a3*b3
# 第一维：0.43 × 0.55 = 0.2365
# 第二维：0.15 × 0.87 = 0.1305
# 第三维：0.89 × 0.66 = 0.5874
# 同方向 → 值大
# 反方向 → 负数
# 垂直 → 0
for i, x_i in enumerate(inputs):
    # dot product = 相似度
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

# ---------------------------------------
# 手动计算一次 dot product（用于验证）
# ---------------------------------------
res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]

print(res)

# 与torch.dot结果比较
print(torch.dot(inputs[0], query))

# ---------------------------------------
# 方法1：简单归一化（不是真正的attention）
# ---------------------------------------

# 把所有score除以总和
# 得到一个权重分布
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

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

# ---------------------------------------
# 方法3：PyTorch官方softmax（推荐）
# ---------------------------------------

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# ---------------------------------------
# 计算 Context Vector（上下文向量）
# ---------------------------------------

query = inputs[1] # journey

# 初始化 context 向量，长度：3 的vector
context_vec_2 = torch.zeros(query.shape)

# 按照注意力权重，对所有token进行加权求和
# 等价于 context_vec_2 = attn_weights @ inputs
# context vector（上下文向量） 融合了全句信息的词表示
# 核心作用：让词“理解上下文”
# 权重：[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(context_vec_2)