import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 存储输入序列（模型输入）
        self.input_ids = []
        # 存储目标序列（模型预测目标）
        self.target_ids = []

        # 使用 tokenizer 将整个文本转换为 token id 序列
        # allowed_special 表示允许特殊 token <|endoftext|>
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 确保 token 数量至少大于 max_length
        # 因为 target 需要比 input 多一个 token
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # 使用滑动窗口（sliding window）从 token 序列中提取训练样本
        # stride 控制窗口移动步长
        # max_length 控制每个训练序列长度
        for i in range(0, len(token_ids) - max_length, stride):

            # 输入序列，例如：
            # [t1 t2 t3 t4]
            input_chunk = token_ids[i:i + max_length]

            # 目标序列（右移一位）：
            # [t2 t3 t4 t5]
            # GPT 的训练目标是预测“下一个 token”
            target_chunk = token_ids[i + 1: i + max_length + 1]

            # 转成 PyTorch tensor 并保存
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # 返回数据集大小（样本数量）
    def __len__(self):
        return len(self.input_ids)

    # 根据索引获取一条数据
    # DataLoader 会调用这个函数
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# 创建 DataLoader 的函数
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # 初始化 GPT-2 tokenizer（tiktoken 提供）
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建自定义数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建 PyTorch DataLoader
    # DataLoader 负责：
    # 1. batch 组合
    # 2. 数据打乱
    # 3. 多线程加载
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 每个 batch 的样本数量
        shuffle=shuffle,        # 是否打乱数据
        drop_last=drop_last,    # 如果最后 batch 不够 batch_size 是否丢弃
        num_workers=num_workers # 数据加载线程数
    )

    return dataloader



vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 如果我们从数据加载器中采样数据，我们会将每个批次中的标记嵌入到一个 256 维向量中。
# 如果批次大小为 8，每个批次包含 4 个标记，则会产生一个 8 x 4 x 256 的张量：
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)


token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# GPT-2 使用绝对位置嵌入，所以我们只需创建另一个嵌入层
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)
# 创建 LLM 中使用的输入嵌入，我们只需将词元和位置嵌入相加
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)