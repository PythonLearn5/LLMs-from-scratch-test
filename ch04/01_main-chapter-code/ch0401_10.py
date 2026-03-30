import torch
import torch.nn as nn

# 残差连接可以有效防止梯度消失，让深层网络更容易训练
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut  # 是否使用“残差连接”（shortcut）

        # 使用 ModuleList 存储多层网络（每一层：Linear + GELU）
        self.layers = nn.ModuleList([
            # 第1层：输入维度 -> 下一层维度
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            # 第2层
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            # 第3层
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            # 第4层
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            # 第5层
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        # 逐层前向传播
        for layer in self.layers:
            # 当前层输出
            layer_output = layer(x)

            # 判断是否可以使用“残差连接”
            # 条件：
            # 1. 开启 shortcut
            # 2. 输入和输出 shape 一样（才能相加）
            if self.use_shortcut and x.shape == layer_output.shape:
                # 残差连接（核心）：x + f(x)
                x = x + layer_output
            else:
                # 普通前向传播
                x = layer_output

        return x

# 用概率方式平滑地决定一个值该保留多少
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU 激活函数（近似公式）
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


def print_gradients(model, x):
    # 前向传播
    output = model(x)

    # 目标值（这里是一个简单的0）
    target = torch.tensor([[0.]])

    # 使用均方误差损失
    loss = nn.MSELoss()
    loss = loss(output, target)

    # 反向传播（计算梯度）
    loss.backward()

    # 遍历所有参数
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 打印每一层权重的梯度平均值（绝对值）
            # 👉 用来观察梯度是否消失
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


# 每一层的维度（共5层）
layer_sizes = [3, 3, 3, 3, 3, 1]

# 输入数据（一个样本，3维）
sample_input = torch.tensor([[1., 0., -1.]])


# ===============================
# ❌ 不使用残差连接
# ===============================
torch.manual_seed(123)  # 固定随机数，保证对比公平

model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)

# 打印梯度
print_gradients(model_without_shortcut, sample_input)

print("==============================================================")
# ===============================
# ✅ 使用残差连接
# ===============================
torch.manual_seed(123)  # 同样初始化（保证对比一致）

model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)

# 打印梯度
print_gradients(model_with_shortcut, sample_input)