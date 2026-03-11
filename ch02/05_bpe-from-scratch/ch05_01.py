text = "This is some text"
# 将文本转换为字节数组（毕竟 BPE 代表的是“字节”对编码）
byte_ary = bytearray(text, "utf-8")
print(byte_ary)
# 当我们调用list()一个bytearray对象时，每个字节都被视为一个单独的元素，结果是一个与字节值对应的整数列表：
ids = list(byte_ary)
print(ids)

# 这是一种将文本转换为 LLM 嵌入层所需的标记 ID 表示的有效方法。
# 然而，这种方法的缺点是它会为每个字符创建一个 ID（对于一篇短文来说，这会产生大量的 ID！）
# 也就是说，对于 17 个字符的输入文本，我们必须使用 17 个标记 ID 作为 LLM 的输入：
print("Number of characters:", len(text))
print("Number of token IDs:", len(ids))