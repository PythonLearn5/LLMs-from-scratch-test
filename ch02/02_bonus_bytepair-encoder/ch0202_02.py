import tiktoken

# 使用来自tiktoken  字节对编码 (BPE) 实现方式
tik_tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, world. Is this-- a test?"

integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

strings = tik_tokenizer.decode(integers)

print(strings)

print(tik_tokenizer.n_vocab)