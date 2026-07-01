import tiktoken
import torch
import torch.nn as nn

# tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

raw_text_temp = "I love AI"

# encode
enc = tokenizer.encode(raw_text_temp)
print("GPT2 tokens:", enc)

# ---------
# 关键：压缩到 0~5
# ---------
unique_tokens = list(dict.fromkeys(enc))[:6]  # 保序去重

token_map = {t: i for i, t in enumerate(unique_tokens)}

mapped_ids = [token_map[t] for t in enc]

print("mapped ids:", mapped_ids)

# embedding
embedding = nn.Embedding(6, 3)

input_ids = torch.tensor(mapped_ids)

out = embedding(input_ids)

print("shape:", out.shape)
print(out)
# decode token id
token_to_small = {t: i for i, t in enumerate(unique_tokens)}
small_to_token = {i: t for i, t in enumerate(unique_tokens)}

vecs = embedding(input_ids)

# ---- decode ----
def decode_from_embedding(vecs, embedding, small_to_token):
    decoded_tokens = []

    weight = embedding.weight  # [6,3]

    for v in vecs:
        # 计算相似度（点积）
        scores = weight @ v

        # 找最相似 token
        idx = torch.argmax(scores).item()

        # 映射回 GPT2 token id
        token_id = small_to_token[idx]

        decoded_tokens.append(token_id)

    return decoded_tokens

decoded_token_ids = decode_from_embedding(vecs, embedding, small_to_token)

print("decoded token ids:", decoded_token_ids)

text = tokenizer.decode(decoded_token_ids)

print("decoded text:", text)