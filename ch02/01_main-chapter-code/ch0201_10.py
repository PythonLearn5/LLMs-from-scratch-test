import torch
import torch.nn as nn

vocab_size = 10000
embedding_dim = 4
max_length = 10

# token embedding
token_embedding = nn.Embedding(vocab_size, embedding_dim)

# position embedding
pos_embedding = nn.Embedding(max_length, embedding_dim)

# 输入 token
input_ids = torch.tensor([40,3021,9552])

# 生成位置
positions = torch.arange(len(input_ids))

# 获取 embedding
token_vec = token_embedding(input_ids)
pos_vec = pos_embedding(positions)

# 最终输入
input_embedding = token_vec + pos_vec

print(input_embedding)

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