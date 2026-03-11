import tiktoken

gpt2_tokenizer = tiktoken.get_encoding("gpt2")
gpt2_tokenizer.encode("This is some text")
# prints [1212, 318, 617, 2420]