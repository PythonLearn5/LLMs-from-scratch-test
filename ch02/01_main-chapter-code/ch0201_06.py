# GPT-2 使用字节对编码 (BPE) 作为其分词器
# 允许模型将不在预定义词汇表中的单词分解成更小的子词单元，甚至是单个字符，从而使其能够处理超出词汇表范围的单词。
# 使用 OpenAI 开源tiktoken库中的 BPE 分词器，该分词器使用 Rust 实现了其核心算法，以提高计算性能。

from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

strings = tokenizer.decode(integers)

print(strings)