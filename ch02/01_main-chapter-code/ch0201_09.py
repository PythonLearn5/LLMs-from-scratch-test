# 输入向量 = token_embedding + position_embedding
# 创建词嵌入
# 使用嵌入层将词元嵌入到连续向量表示中。
# 嵌入层是LLM本身的一部分，并在模型训练期间进行更新（训练）。
# 完整流程是 embedding只是向量，没有顺序概念。 Position Embedding 就是告诉模型：这个词在第几个位置
# 文本
#  ↓
# Tokenizer
#  ↓
# Token IDs
#  ↓
# Token Embedding
#  ↓
# Position Embedding
#  ↓
# 相加
#  ↓
# Transformer
# InputEmbedding（输入向量） = TokenEmbedding + PositionEmbedding
import torch

# 创建一个 token id 序列（例如 tokenizer 输出的 token）
# 这里表示一个长度为4的序列
input_ids = torch.tensor([2, 3, 5, 1])

# 词表大小（vocabulary size）
# 表示总共有多少个不同的 token
vocab_size = 6

# 每个 token 对应的向量维度（embedding dimension）
# 这里每个 token 会被映射为一个 3 维向量
output_dim = 3

# 设置随机种子，使每次运行生成的随机数一致（方便复现结果）
torch.manual_seed(123)

# 创建 embedding 层 token embedding
# 作用：把 token id 映射成向量
# 相当于一个查表操作
# 表大小 6*3 = vocab_size × output_dim
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# PyTorch 会随机初始化 embedding_layer.weight
# print(embedding_layer)

# 查看 embedding 权重矩阵
# 形状: (vocab_size, output_dim)
# 也就是：
# 6 个 token
# 每个 token 一个 3 维向量
print(embedding_layer.weight)

# 输入 token id = 3
# embedding 层会返回权重矩阵的第 3 行
# 相当于：
# embedding_vector = weight[3]
print(embedding_layer(torch.tensor([3])))

# 输入多个 token id
# embedding 会把每个 token 转换为对应向量
# 输入 shape:  (4,)
# 输出 shape:  (4, 3)
# 即 4 个 token → 4 个向量，每个向量 3 维
# 相当于取：2, 3, 5, 1 重新组合一个embedding_layer
print(embedding_layer(input_ids))