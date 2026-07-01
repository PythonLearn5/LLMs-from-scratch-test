import tiktoken
import torch
import torch.nn as nn

enc = tiktoken.get_encoding("gpt2")

# GPT-2 tokenizer 是 byte-level BPE（字节级编码）
# 任何文本都可以先变成 bytes，而 bytes 永远是完整的（0–255）
# 1、任何字符都能变 bytes
# 2、byte 是“封闭集合” bytes 只有：0 ~ 255（固定 256 种）
# 所以不存在：没有字节、无法表示的字符
# 3、BPE 保证“永远可拆”
# 比如
# "hellooo"
# → h e l l o o o
# → BPE拆分
# → 一定能编码

print(enc.encode("hello"))
print(enc.encode("helloo"))
print(enc.encode("hellooo"))
print(enc.encode("你好"))
print(enc.encode("你"))
print(enc.encode("你你"))
print(enc.encode("😀"))