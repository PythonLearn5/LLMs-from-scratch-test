# 实现一个更接近真实 Transformer 的 Attention 计算。
# 和你前一段代码相比，这里已经加入了 Q、K、V（Query / Key / Value）线性变换，这就是 Transformer 和 GPT 中真正使用的注意力结构。
import torch

# --------------------------------------------------
# 输入：6个token，每个token是3维向量
# 实际模型中通常是 768 / 1024 / 4096 维
# --------------------------------------------------

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x1)
   [0.55, 0.87, 0.66], # journey  (x2)
   [0.57, 0.85, 0.64], # starts   (x3)
   [0.22, 0.58, 0.33], # with     (x4)
   [0.77, 0.25, 0.10], # one      (x5)
   [0.05, 0.80, 0.55]] # step     (x6)
)

# --------------------------------------------------
# 选择第2个token（journey）
# --------------------------------------------------

x_2 = inputs[1] # second input element

# 输入向量维度
d_in = inputs.shape[1]  # shape=[6*3] shape[1]=3

# 输出向量维度（attention内部维度）
d_out = 2

# --------------------------------------------------
# 设置随机种子，保证每次运行结果一致
# --------------------------------------------------
torch.manual_seed(123)

# --------------------------------------------------
# 创建三个权重矩阵
# 用来生成 Query / Key / Value
# --------------------------------------------------

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 权重矩阵形状
# (3 × 2)
# --------------------------------------------------
# 第一步：计算 Query / Key / Value
# --------------------------------------------------
print(x_2)     # 选择第2个token（journey=[0.55, 0.87, 0.66]）
print(W_query)
# 当前token的Query
# 其中 @ 代表的是：矩阵乘法（Matrix Multiplication）
# * 对应元素相乘（逐元素） A = [1, 2] B = [3, 4] 则 A * B = [1×3, 2×4] = [3, 8]
# @ 矩阵乘法（线性代数）  @ 是点积（dot product） A @ B = 1×3 + 2×4 = 11  两个向量压成一个数
# @ 把所有特征综合起来打一个分数  * 你和我每一项单独对比，不做综合判断
# [0.5500, 0.8700, 0.6600] @ [[0.2961, 0.5166],[0.2517, 0.6886],[0.0740, 0.8665]]
# 第1列计算 = 0.55×0.2961 + 0.87×0.2517 + 0.66×0.0740 ≈ 0.4306
# 第2列计算 = 0.55×0.5166 + 0.87×0.6886 + 0.66×0.8665 = 0.57189
query_2 = x_2 @ W_query

# 当前token的Key
key_2 = x_2 @ W_key

# 当前token的Value
value_2 = x_2 @ W_value

print(query_2)

# --------------------------------------------------
# 计算所有token的 Key 和 Value
# --------------------------------------------------
# [[0.43, 0.15, 0.89], [0.55, 0.87, 0.66],  [0.57, 0.85, 0.64], [0.22, 0.58, 0.33],  [0.77, 0.25, 0.10], [0.05, 0.80, 0.55]]] @ [[0.1366, 0.1025],[0.1841, 0.7264],[0.3153, 0.6871]]
keys = inputs @ W_key
values = inputs @ W_value

print(keys)
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# 结果
# 6 × 2

# --------------------------------------------------
# 第二步：计算 Attention Score
# --------------------------------------------------

# journey 对自己的 attention score
keys_2 = keys[1] # Python starts index at 0

attn_score_22 = query_2.dot(keys_2)

print(attn_score_22)

# --------------------------------------------------
# 计算 journey 对所有token的 attention score
# --------------------------------------------------

attn_scores_2 = query_2 @ keys.T

print(attn_scores_2)

# 得到：
# 6个注意力分数

# --------------------------------------------------
# 第三步：scaled dot-product attention
# --------------------------------------------------

# key向量维度
d_k = keys.shape[1]

# 注意力权重
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)

print(attn_weights_2)

# --------------------------------------------------
# 第四步：计算 Context Vector
# --------------------------------------------------

context_vec_2 = attn_weights_2 @ values

print(context_vec_2)