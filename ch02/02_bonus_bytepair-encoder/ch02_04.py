# pip install transformers

# import transformers
# print(transformers.__version__)


# from transformers import GPT2Tokenizer
# strings = "Hello, world. Is this-- a test?"
# hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# hf_tokenizer(strings)["input_ids"]

from transformers import GPT2TokenizerFast
strings = "Hello, world. Is this-- a test?"
hf_tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")
hf_tokenizer_fast(strings)["input_ids"]