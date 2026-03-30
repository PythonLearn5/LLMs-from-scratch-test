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


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

# GPT-2 124M 模型的配置参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词表大小（GPT2 BPE tokenizer 的词汇量）
    "context_length": 1024, # 最大上下文长度（最大 token 数）
    "emb_dim": 768,         # token embedding 维度
    "n_heads": 12,          # 多头注意力的头数量
    "n_layers": 12,         # Transformer block 层数
    "drop_rate": 0.1,       # Dropout 比例（训练时随机丢弃神经元）
    "qkv_bias": False       # Q K V 线性层是否使用 bias
}

start_context = "Hello, I am"
tokenizer = tiktoken.get_encoding("gpt2")

encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval() # disable dropout

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)