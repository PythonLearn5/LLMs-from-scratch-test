import torch

print("PyTorch version:", torch.__version__)

# 假设我们有以下 3 个训练样本，
# 它们可能代表 LLM 上下文中的标记 ID
idx = torch.tensor([2, 3, 1])

# 嵌入矩阵的行数可以通过以下方式确定：
# 取最大词元 ID + 1。
# 如果最大词元 ID 为 3，则我们需要 4 行，因为可能的词元 ID 为：
# token IDs 0, 1, 2, 3
num_idx = max(idx)+1

# 期望的嵌入维度是一个超参数
out_dim = 5

# 我们使用随机种子是为了保证结果的可复现性，因为
# 嵌入层中的权重是用
# 小的随机值初始化的
torch.manual_seed(123)

embedding = torch.nn.Embedding(num_idx, out_dim)
# 查看嵌入权重
print(embedding.weight)
# 使用嵌入层来获得 ID 为 1 的训练样本的向量表示
print(embedding(torch.tensor([1])))

# 我们转换之前定义的所有训练样本：就是在原tensor 中查找 2, 3, 1 重新组装
idx = torch.tensor([2, 3, 1])
print(embedding(idx))