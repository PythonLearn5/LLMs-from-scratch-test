# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# 该代码来自 OpenAI GPT-2 官方 tokenizer 实现

# License:
# Modified MIT License
# 说明：MIT许可证，允许自由使用、修改、发布

# Software Copyright (c) 2019 OpenAI

# 以下是许可证声明，说明可以自由使用代码，但不提供任何担保
# 省略解释（只是法律声明）


import os
import json
import regex as re   # 使用 regex 库（比 Python 标准 re 更强大，支持 Unicode 特性）
import requests
from tqdm import tqdm
from functools import lru_cache


@lru_cache()
def bytes_to_unicode():
    """
    创建 UTF-8 字节 → Unicode 字符 的映射表

    BPE算法是在 Unicode 字符串上操作的，
    但输入文本实际是 UTF-8 字节流。

    这个函数的作用是：
    将 0-255 的字节映射为可打印的 Unicode 字符，
    从而避免控制字符或空白字符影响 BPE 处理。
    """

    # bs = byte sequence
    # 先选取可打印字符范围
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))

    # cs = unicode characters
    cs = bs[:]

    n = 0

    # 遍历所有 256 个字节
    for b in range(2**8):

        # 如果该字节不在已选字符范围内
        if b not in bs:
            bs.append(b)

            # 映射到新的 Unicode 位置
            cs.append(2**8 + n)

            n += 1

    # 转换为 Unicode 字符
    cs = [chr(n) for n in cs]

    # 返回字节 → Unicode 映射
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    获取一个 token 中所有相邻字符对（bigram）

    输入：
        ('l','o','w')

    输出：
        {('l','o'), ('o','w')}
    """

    pairs = set()

    # 前一个字符
    prev_char = word[0]

    # 遍历后续字符
    for char in word[1:]:

        # 添加字符对
        pairs.add((prev_char, char))

        prev_char = char

    return pairs


class Encoder:

    def __init__(self, encoder, bpe_merges, errors="replace"):

        # token → id 的词表
        self.encoder = encoder

        # id → token 的反向词表
        self.decoder = {v: k for k, v in self.encoder.items()}

        # decode 错误处理方式
        self.errors = errors

        # byte → unicode 映射
        self.byte_encoder = bytes_to_unicode()

        # unicode → byte 映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # BPE merge 规则优先级
        # 每个 pair 对应一个 rank（越小优先级越高）
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        # 缓存 BPE 结果（加速）
        self.cache = {}

        # 正则表达式，用于初步分词
        # 匹配：
        # 单词
        # 数字
        # 标点
        # 空格
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|
            \p{L}+|
            \p{N}+|
            [^\s\p{L}\p{N}]+|
            \s+"""
        )


    def bpe(self, token):

        # 如果已经计算过，直接返回缓存
        if token in self.cache:
            return self.cache[token]

        # 将 token 转为字符 tuple
        word = tuple(token)

        # 获取所有字符对
        pairs = get_pairs(word)

        if not pairs:
            return token

        # BPE合并循环
        while True:

            # 找到 rank 最小的字符对（优先合并）
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )

            # 如果该字符对不在 merge 规则中，则停止
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram

            new_word = []
            i = 0

            # 合并字符
            while i < len(word):

                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j

                except ValueError:
                    new_word.extend(word[i:])
                    break

                # 如果匹配到 pair
                if word[i] == first and i < len(word)-1 and word[i+1] == second:

                    # 合并
                    new_word.append(first+second)
                    i += 2

                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)

            word = new_word

            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # 用空格分隔 token
        word = " ".join(word)

        # 缓存结果
        self.cache[token] = word

        return word


    def encode(self, text):

        # 最终 token id
        bpe_tokens = []

        # 正则分词
        for token in re.findall(self.pat, text):

            # 将 token 转为 UTF-8 bytes
            # 再映射为 Unicode 字符
            token = "".join(
                self.byte_encoder[b]
                for b in token.encode("utf-8")
            )

            # 执行 BPE
            bpe_tokens.extend(
                self.encoder[bpe_token]
                for bpe_token in self.bpe(token).split(" ")
            )

        return bpe_tokens


    def decode(self, tokens):

        # token id → token
        text = "".join([self.decoder[token] for token in tokens])

        # Unicode → byte
        text = bytearray(
            [self.byte_decoder[c] for c in text]
        ).decode("utf-8", errors=self.errors)

        return text


def get_encoder(model_name, models_dir):

    # 读取词表
    with open(os.path.join(models_dir, model_name, "encoder.json"), "r") as f:
        encoder = json.load(f)

    # 读取 BPE merge 规则
    with open(os.path.join(models_dir, model_name, "vocab.bpe"), "r", encoding="utf-8") as f:
        bpe_data = f.read()

    # 解析 merge 规则
    bpe_merges = [
        tuple(merge_str.split())
        for merge_str in bpe_data.split("\n")[1:-1]
    ]

    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():

    # GPT-2 tokenizer 文件目录
    subdir = "gpt2_model"

    # 如果目录不存在则创建
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    # Windows路径处理
    subdir = subdir.replace("\\", "/")

    # 下载两个文件
    for filename in ["encoder.json", "vocab.bpe"]:

        r = requests.get(
            "https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename,
            stream=True
        )

        with open(os.path.join(subdir, filename), "wb") as f:

            # 获取文件大小
            file_size = int(r.headers["content-length"])

            chunk_size = 1000

            # 进度条显示下载
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True
            ) as pbar:

                # 分块下载
                for chunk in r.iter_content(chunk_size=chunk_size):

                    f.write(chunk)

                    pbar.update(chunk_size)