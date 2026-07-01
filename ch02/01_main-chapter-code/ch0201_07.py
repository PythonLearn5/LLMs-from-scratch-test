from importlib.metadata import version
import tiktoken
import torch
import torch.nn as nn

print("tiktoken version:", version("tiktoken"))

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

# 简单测试
raw_text_temp = "I love AI"
enc_text_temp = tokenizer.encode(raw_text_temp)
# 从第 0 个 token 开始打印后面的 3 个token
print(enc_text_temp[0:3])
print(tokenizer.decode(enc_text_temp[0:1]))
# GPT-2 的 tokenizer 把“空格”当成词的一部分编码了，而不是分隔符。 所以前面会有空格
print(tokenizer.decode([1842]))
print(tokenizer.decode([9552]))

# 把“整数 token id” → “高维向量” --> Embedding
embedding = nn.Embedding(num_embeddings=50257, embedding_dim=768)

# 先将tokenId 转换成张量
tokenOne = torch.tensor([40])
tokens = torch.tensor([40, 1842, 995])
# 本质：一个查表操作（不是计算）
# Embedding 不是复杂公式，而是：
# 一个巨大的矩阵 + 查表（lookup）
# 出现在相似上下文的 token，会被优化成相似向量
# 初始语义表示 X
# X → Q, K, V 模型真正的“理解能力”是在 attention 里形成的
outOne = embedding(tokenOne)
outAll = embedding(tokens)
print(outOne.shape)
print(outAll.shape)
# print(outOne)
# print(outAll)

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]

context_size = 4
# 由于我们希望模型预测下一个单词，因此目标值是将输入值向右平移一位。
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")
# 逐一来看，预测结果如下：
# married a rich widow, and established himself in a villa on the Riviera.
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))