import re

text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
# 标点符号 或 --  或 空白字符
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(preprocessed[:30])
# 基于这些词元，我们现在可以构建一个包含所有唯一词元的词汇表
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)
print(all_words)

vocab = {token:integer for integer,token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break