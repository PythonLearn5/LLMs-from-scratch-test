# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# 本文件汇总了第2-4章实现 GPT 所需的全部核心代码
# 可以直接作为脚本运行

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#####################################
# Chapter 2
#####################################


# 数据集类：把长文本切分成 GPT 训练所需的输入/目标序列
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 使用 tokenizer 把整个文本转换成 token id
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口切分文本
        # 每次窗口移动 stride 个 token
        for i in range(0, len(token_ids) - max_length, stride):

            # 输入序列
            input_chunk = token_ids[i:i + max_length]

            # 目标序列（右移1位）
            # GPT 的训练目标：预测下一个 token
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # 返回数据集长度
    def __len__(self):
        return len(self.input_ids)

    # 获取指定索引的数据
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# 创建 DataLoader
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):

    # 初始化 GPT2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建 Dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################


# 多头注意力机制 (Multi Head Attention)
class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # 确保输出维度可以被 head 数整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads

        # 每个 head 的维度
        self.head_dim = d_out // num_heads

        # Q K V 投影矩阵
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 最终输出投影
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        # 创建 causal mask
        # 上三角矩阵，用于防止看到未来 token
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):

        # x 形状
        # (batch, tokens, embedding_dim)
        b, num_tokens, d_in = x.shape

        # 计算 Q K V
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 拆分多头
        # (b, tokens, d_out)
        # ->
        # (b, tokens, heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 调整维度顺序
        # (b, heads, tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算 attention score
        # Q @ K^T
        attn_scores = queries @ keys.transpose(2, 3)

        # 取出 mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 将未来位置填充为 -inf
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scaled Dot Product Attention
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )

        attn_weights = self.dropout(attn_weights)

        # Attention * Value
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并 heads
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 最终投影
        context_vec = self.out_proj(context_vec)

        return context_vec


#####################################
# Chapter 4
#####################################


# Layer Normalization
class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5

        # 可学习参数
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):

        # 计算均值
        mean = x.mean(dim=-1, keepdim=True)

        # 计算方差
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 标准化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # scale + shift
        return self.scale * norm_x + self.shift


# GELU 激活函数
class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        # GPT 使用的 GELU 近似公式
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            )
        )


# Transformer 前馈网络
class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(

            # 维度扩大4倍
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),

            GELU(),

            # 再缩回 emb_dim
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# Transformer Block
class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # 多头注意力
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        # 前馈网络
        self.ff = FeedForward(cfg)

        # 两个 LayerNorm
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        # ===== Attention Block =====

        shortcut = x

        x = self.norm1(x)

        x = self.att(x)

        x = self.drop_shortcut(x)

        # 残差连接
        x = x + shortcut

        # ===== FeedForward Block =====

        shortcut = x

        x = self.norm2(x)

        x = self.ff(x)

        x = self.drop_shortcut(x)

        # 残差连接
        x = x + shortcut

        return x


# GPT 模型
class GPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # token embedding
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"]
        )

        # position embedding
        self.pos_emb = nn.Embedding(
            cfg["context_length"],
            cfg["emb_dim"]
        )

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最后 layernorm
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # 输出层
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False
        )

    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape

        # token embedding
        tok_embeds = self.tok_emb(in_idx)

        # position embedding
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # token + position
        x = tok_embeds + pos_embeds

        x = self.drop_emb(x)

        # transformer
        x = self.trf_blocks(x)

        x = self.final_norm(x)

        # 输出 logits
        logits = self.out_head(x)

        return logits


# 文本生成函数
def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):

        # 截取 context window
        idx_cond = idx[:, -context_size:]

        # 前向推理
        with torch.no_grad():
            logits = model(idx_cond)

        # 取最后一个 token 的 logits
        logits = logits[:, -1, :]

        # greedy decoding
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 拼接 token
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():

    # GPT-2 small 配置
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    torch.manual_seed(123)

    # 初始化模型
    model = GPTModel(GPT_CONFIG_124M)

    model.eval()

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")

    # 编码
    encoded = tokenizer.encode(start_context)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print("Input text:", start_context)
    print("Encoded:", encoded)

    # 生成文本
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    # 解码
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()