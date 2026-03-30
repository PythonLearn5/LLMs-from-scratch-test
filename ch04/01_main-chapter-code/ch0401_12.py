import torch
import torch.nn as nn
from previous_chapters import MultiHeadAttention
import tiktoken
# 一个完整的 GPT 前向传播：文本 → embedding → Transformer → 预测下一个词
class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5  # 防止除0
    # 可学习参数：缩放（γ）
    self.scale = nn.Parameter(torch.ones(emb_dim))
    # 可学习参数：平移（β）
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    # x.shape = [batch, seq, emb_dim]

    # 对最后一维（embedding维）求均值
    mean = x.mean(dim=-1, keepdim=True)

    # 对最后一维求方差
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # 标准化
    norm_x = (x - mean) / torch.sqrt(var + self.eps)

    # 再做线性变换（可学习）
    return self.scale * norm_x + self.shift

# GELU 激活函数（Transformer 标配）
class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    # 近似公式（比 ReLU 更平滑）
    return 0.5 * x * (1 + torch.tanh(
      torch.sqrt(torch.tensor(2.0 / torch.pi)) *
      (x + 0.044715 * torch.pow(x, 3))
    ))

# 前馈网络（FFN）
class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    # 典型结构：Linear → GELU → Linear
    self.layers = nn.Sequential(
      # 先扩展维度（通常 ×4）
      nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
      GELU(),
      # 再压回原维度
      nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
    )

  def forward(self, x):
    return self.layers(x)

# Transformer Block（核心模块）
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
            qkv_bias=cfg["qkv_bias"])

        # 前馈网络
        self.ff = FeedForward(cfg)

        # 两个 LayerNorm（分别用于 Attention 和 FFN）
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # 用在残差连接上的 dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # =========================
        # 1️⃣ Attention Block
        # =========================

        shortcut = x  # 保存输入（用于残差连接）

        x = self.norm1(x)      # 先做 LayerNorm（Pre-LN结构）
        x = self.att(x)        # 多头注意力
        x = self.drop_shortcut(x)  # dropout

        x = x + shortcut  # 残差连接（关键！） 防止梯度消失、允许信息“直通”

        # =========================
        # 2️⃣ FeedForward Block
        # =========================

        shortcut = x  # 再保存一次

        x = self.norm2(x)  # 再做 LayerNorm
        x = self.ff(x)     # 前馈网络
        x = self.drop_shortcut(x)

        x = x + shortcut  # 残差连接

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # =========================
        # 1️⃣ Token Embedding（词向量）
        # =========================
        # 把 token id → 向量
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # =========================
        # 2️⃣ Position Embedding（位置编码）
        # =========================
        # 表示每个 token 在序列中的位置
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # embedding 后做 dropout（防过拟合）
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # =========================
        # 3️⃣ Transformer Blocks（核心）
        # =========================
        # 堆叠多个 TransformerBlock（GPT-2 是 12 层）
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # =========================
        # 4️⃣ 最终 LayerNorm
        # =========================
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # =========================
        # 5️⃣ 输出层（映射回词表）
        # =========================
        # 把 embedding → vocab_size（预测下一个词）
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        # in_idx.shape = [batch_size, seq_len]
        batch_size, seq_len = in_idx.shape

        # =========================
        # 1️⃣ token embedding
        # =========================
        tok_embeds = self.tok_emb(in_idx)
        # shape: [batch, seq, emb_dim]

        # =========================
        # 2️⃣ position embedding
        # =========================
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        # shape: [seq_len, emb_dim]

        # =========================
        # 3️⃣ token + position
        # =========================
        x = tok_embeds + pos_embeds
        # broadcasting → [batch, seq, emb_dim]

        # dropout
        x = self.drop_emb(x)

        # =========================
        # 4️⃣ Transformer blocks
        # =========================
        x = self.trf_blocks(x)
        # shape 不变：[batch, seq, emb_dim]

        # =========================
        # 5️⃣ final layer norm
        # =========================
        x = self.final_norm(x)

        # =========================
        # 6️⃣ 输出 logits
        # =========================
        logits = self.out_head(x)
        # shape: [batch, seq, vocab_size]

        return logits
# GPT-2 124M 模型的配置参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词表大小
    "context_length": 1024, # 最大序列长度
    "emb_dim": 768,         # embedding维度
    "n_heads": 12,          # 注意力头数
    "n_layers": 12,         # Transformer层数
    "drop_rate": 0.1,       # dropout比例
    "qkv_bias": False       # QKV是否加bias
}

# 构造 batch
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# 使用 GPT2 tokenizer（把文本 → token id）
tokenizer = tiktoken.get_encoding("gpt2")

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

# stack 成 batch
batch = torch.stack(batch, dim=0)
# shape: [2, seq_len]

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# 假设 out.shape = [batch, seq_len, vocab_size]
# 方法1：取每个位置最大概率对应的 token id
pred_ids = torch.argmax(out, dim=-1)
# pred_ids.shape = [batch, seq_len]

# tokenizer = tiktoken.get_encoding("gpt2")
for i, seq in enumerate(pred_ids):
    text = tokenizer.decode(seq.tolist())  # 转成普通字符串
    print(f"Sample {i}: {text}")

# Softmax 转概率
probs = torch.softmax(out, dim=-1)  # [batch, seq_len, vocab_size]

# 采样一个 token
sampled_ids = torch.multinomial(probs[0, -1], num_samples=1)
print(sampled_ids)