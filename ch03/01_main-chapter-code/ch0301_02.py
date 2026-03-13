# 注意力机制（Attention） 是现代大模型（GPT、BERT、Transformer）的核心机制
# 作用：当模型处理某个词时，决定应该关注句子中的哪些词

import torch

# 6个token，每个token用3维向量表示
# 实际大模型中通常是768维 / 1024维
# 这里为了教学演示，只使用3维向量

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x1)
   [0.55, 0.87, 0.66], # journey  (x2)
   [0.57, 0.85, 0.64], # starts   (x3)
   [0.22, 0.58, 0.33], # with     (x4)
   [0.77, 0.25, 0.10], # one      (x5)
   [0.05, 0.80, 0.55]] # step     (x6)
)

# --------------------------------------------------
# 第一步：计算 Attention Score
# 每个token与所有token计算相似度
# --------------------------------------------------

# 创建一个 6x6 的矩阵存储注意力分数
# 行 = 当前token
# 列 = 被关注的token
attn_scores = torch.empty(6, 6)

# 使用两层循环计算 dot product
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        # 点积表示两个向量的相似度
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# --------------------------------------------------
# 上面的循环其实可以用矩阵乘法一次完成
# 这是Transformer真正使用的方式
# --------------------------------------------------

# inputs @ inputs.T
# 相当于所有token互相计算dot product
attn_scores = inputs @ inputs.T

print(attn_scores)

# --------------------------------------------------
# 第二步：softmax归一化
# 把相似度转换成注意力权重
# --------------------------------------------------

# dim=-1 表示按每一行做softmax
# 每一行表示：当前token关注其他token的概率
attn_weights = torch.softmax(attn_scores, dim=-1)

print(attn_weights)

# --------------------------------------------------
# 验证某一行的softmax是否等于1
# --------------------------------------------------

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)

# 所有行的和都应该等于1
print("All row sums:", attn_weights.sum(dim=-1))

# --------------------------------------------------
# 第三步：计算 Context Vector
# --------------------------------------------------

# 用注意力权重加权原始输入向量
# 公式：
# Context = AttentionWeights × Inputs

all_context_vecs = attn_weights @ inputs

print(all_context_vecs)