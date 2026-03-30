import torch
import torch.nn as nn
from previous_chapters import MultiHeadAttention  # 多头注意力（你前面实现的）
# Transformer 的一个标准 Block（Attention + FFN + LayerNorm + 残差）
# 一个标准 Transformer Block
# x
#  ↓
# LayerNorm
#  ↓
# Multi-Head Attention
#  ↓
# Dropout
#  ↓
# + 残差
#  ↓
# LayerNorm
#  ↓
# FeedForward（MLP）
#  ↓
# Dropout
#  ↓
# + 残差
#  ↓
# 输出
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

torch.manual_seed(123)

# 输入数据
x = torch.rand(2, 4, 768)  # [batch=2, token数=4, emb_dim=768]

# 构建一个 Transformer Block
block = TransformerBlock(GPT_CONFIG_124M)

# 前向传播
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)