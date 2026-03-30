import torch.nn as nn
import torch

input_ids = torch.tensor([
  [1, 2, 3, 4, 5, 6]
], dtype=torch.long)

vocab_size = 10000
embedding_dim = 4

# Embedding  本质是一个查表操作
embedding = nn.Embedding(vocab_size, embedding_dim)
dropout = nn.Dropout(p=0.1)

x = embedding(input_ids)
x = dropout(x)

print(x.shape)  # [1, 6, 4]
print(x)