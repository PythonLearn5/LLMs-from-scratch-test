# pip install transformers

# import transformers
# print(transformers.__version__)


# from transformers import GPT2Tokenizer
# strings = "Hello, world. Is this-- a test?"
# hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# hf_tokenizer(strings)["input_ids"]
# 导入 Hugging Face 提供的 快速版 GPT-2 Tokenizer 通过 Hugging Face transformers 使用 BPE
from transformers import GPT2TokenizerFast
strings = "Hello, world. Is this-- a test?"
# 加载 GPT-2 Tokenizer
# 第一次运行时会下载：
#
# vocab.json
# merges.txt
# tokenizer_config.json
#
# 这些文件来自 GPT-2 官方模型。
hf_tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")
# 开始编码
list = hf_tokenizer_fast(strings)
input_ids = list["input_ids"]
print(input_ids)


# 解码
decoded = hf_tokenizer_fast.decode(input_ids)
print(decoded)