import tiktoken
gpt2_tokenizer = tiktoken.get_encoding("gpt2")

for i in range(300):
    decoded = gpt2_tokenizer.decode([i])
    print(f"{i}: {decoded}")
"""
prints:
0: !
1: "
2: #
...
255: �  # <---- single character tokens up to here
256:  t
257:  a
...
298: ent
299:  n
"""