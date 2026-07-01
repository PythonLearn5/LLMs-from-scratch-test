import tiktoken

gpt2_tokenizer = tiktoken.get_encoding("gpt2")

integers = gpt2_tokenizer.encode("This is some text")  # , allowed_special={"<|endoftext|>"}
print(integers)
# prints [1212, 318, 617, 2420]