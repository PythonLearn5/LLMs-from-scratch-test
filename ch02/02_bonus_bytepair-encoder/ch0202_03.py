# 使用 GPT-2 中使用的原始 BPE 实现
from bpe_openai_gpt2 import get_encoder, download_vocab

download_vocab()

orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")

text = "Hello, world. Is this-- a test?"

integers = orig_tokenizer.encode(text)

print(integers)

strings = orig_tokenizer.decode(integers)

print(strings)