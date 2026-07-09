import torch
import torch.nn.functional as F

# 创建一个1×6张量
#
# 注意这里指定 float32
# 因为 GELU 只能处理浮点数
#
y = torch.tensor([
    [1, 2, 3, 0, 5, -6]
], dtype=torch.float32)

# GELU（Gaussian Error Linear Unit）
# 它不像ReLU那样：
# 小于0直接变0
# 而是：
#
# 根据数值大小"平滑"决定保留多少
# 数值越大 保留越多
# 数值越小 抑制越多
# 因此GELU输出更加连续
#
y = F.gelu(y)

print("\nGELU 输出：")
print(y)

