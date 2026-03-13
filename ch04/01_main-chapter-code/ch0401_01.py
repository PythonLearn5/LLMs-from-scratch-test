# 从 importlib.metadata 中导入 version 方法，用于获取库的版本号
from importlib.metadata import version

# 导入 PyTorch 主库
import torch

# 导入 PyTorch 的神经网络模块
import torch.nn as nn

# GPT-2 使用的 BPE tokenizer 实现
import tiktoken


# 打印 matplotlib 库版本
print("matplotlib version:", version("matplotlib"))

# 打印 torch 版本
print("torch version:", version("torch"))

# 打印 tiktoken 版本
print("tiktoken version:", version("tiktoken"))


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


# 定义一个简化版 GPT 模型（只是结构示例，并没有真正实现 Transformer）
class DummyGPTModel(nn.Module):

    # 构造函数
    def __init__(self, cfg):
        super().__init__()

        # token embedding
        # 把 token id 映射为 emb_dim 维向量
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # 位置 embedding
        # 用来表示 token 在序列中的位置
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # embedding dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks（这里只是占位符）
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最后的 LayerNorm（占位）
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # 输出层
        # 把 embedding 映射回 vocab_size，用于预测下一个 token
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    # 前向传播
    def forward(self, in_idx):

        # in_idx shape:
        # (batch_size, seq_len)
        batch_size, seq_len = in_idx.shape

        # token embedding
        # (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)

        # 创建位置 index: [0,1,2,...seq_len-1]
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # token embedding + position embedding
        x = tok_embeds + pos_embeds

        # dropout
        x = self.drop_emb(x)

        # 通过 Transformer blocks
        x = self.trf_blocks(x)

        # LayerNorm
        x = self.final_norm(x)

        # 输出 logits
        # (batch_size, seq_len, vocab_size)
        logits = self.out_head(x)

        return logits


# TransformerBlock 占位实现
class DummyTransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # 这里只是占位，没有真正实现注意力机制

    def forward(self, x):
        # 直接返回输入（什么都没做）
        return x


# LayerNorm 占位实现
class DummyLayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        # 参数只是为了模拟 LayerNorm 接口
        # 实际没有实现

    def forward(self, x):

        # 什么都不做，直接返回输入
        return x



# 获取 GPT2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 构造 batch
batch = []

# 两个示例句子
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# tokenizer.encode -> 把文本转成 token id
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

# stack 成 tensor
# shape: (2, seq_len)
batch = torch.stack(batch, dim=0)

print(batch)


# 设置随机种子（保证结果可复现）
torch.manual_seed(123)

# 创建 GPT 模型
model = DummyGPTModel(GPT_CONFIG_124M)

# 前向传播
logits = model(batch)

# 输出 shape
print("Output shape:", logits.shape)

# 输出 logits
print(logits)


# 再次设置随机种子
torch.manual_seed(123)

# 创建一个示例 batch
# 2 个样本，每个样本 5 个特征
batch_example = torch.randn(2, 5)

# 定义一个简单神经网络
layer = nn.Sequential(
    nn.Linear(5, 6),  # 5 -> 6
    nn.ReLU()         # 激活函数
)

# 前向传播
out = layer(batch_example)

print(out)


# 计算每一行的均值
mean = out.mean(dim=-1, keepdim=True)

# 计算每一行的方差
var = out.var(dim=-1, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)