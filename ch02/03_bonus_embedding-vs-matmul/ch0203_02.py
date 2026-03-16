import torch

print("PyTorch version:", torch.__version__)

idx = torch.tensor([2, 3, 1])
num_idx = max(idx)+1
out_dim = 5

# 将令牌 ID 转换为独热编码表示
onehot = torch.nn.functional.one_hot(idx)
print(onehot)
# 初始化一个Linear层，该层执行矩阵乘法
torch.manual_seed(123)
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
print(linear.weight)

# PyTorch 中的线性层也使用较小的随机权重进行初始化；为了将其与Embedding上面的层直接比较，我们必须使用相同的较小随机权重，这就是为什么我们在这里重新分配它们的原因：
embedding = torch.nn.Embedding(num_idx, out_dim)
linear.weight = torch.nn.Parameter(embedding.weight.T)

linear(onehot.float())

print(embedding(idx))