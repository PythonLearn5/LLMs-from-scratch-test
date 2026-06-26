import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# 打印当前 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 自定义数据集类，用于从文本中生成 GPT 训练需要的 (input, target) 数据对
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
# shuffle 是否打乱数据
# drop_last 如果最后 batch 不够 batch_size 是否丢弃
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


# 读取文本文件（训练数据）
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# 创建 dataloader
# max_length=4 表示每个训练样本长度为 4
# stride=1 表示窗口每次移动 1 个 token（高度重叠）
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

# 创建 dataloader 迭代器
data_iter = iter(dataloader)

# 获取第一个 batch
first_batch = next(data_iter)
print(first_batch)

# 获取第二个 batch
second_batch = next(data_iter)
print(second_batch)


# 再创建一个 dataloader
# stride=4 表示窗口每次移动 4 个 token（不重叠）
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

# 创建迭代器
data_iter = iter(dataloader)

# 获取一个 batch
inputs, targets = next(data_iter)

# 打印输入序列
print("Inputs:\n", inputs)

# 打印目标序列（模型要预测的 token）
print("\nTargets:\n", targets)