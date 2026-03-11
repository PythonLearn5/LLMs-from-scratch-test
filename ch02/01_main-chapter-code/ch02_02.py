with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
# 打印字符串 raw_text 的前 99 个字符
print(raw_text[:99])