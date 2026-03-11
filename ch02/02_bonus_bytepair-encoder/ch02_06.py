# test time
from bpe_openai_gpt2 import get_encoder, download_vocab
import time

download_vocab()

start = time.time()
orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")

with open("../01_main-chapter-code/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

integers = orig_tokenizer.encode(raw_text)

print(time.time() - start)


start = time.time()
strings = orig_tokenizer.decode(integers)
print(time.time() - start)


