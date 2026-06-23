import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        # 单词--token_ID
        self.str_to_int = vocab
        # token_ID--单词
        self.int_to_str = {i: s for s, i in vocab.items()}
        print(self.int_to_str)

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 删除标点符号前面多余的空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)
print(all_words)

vocab = {token:integer for integer,token in enumerate(all_words)}


tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

tokenizer.decode(ids)

tokenizer.decode(tokenizer.encode(text))
