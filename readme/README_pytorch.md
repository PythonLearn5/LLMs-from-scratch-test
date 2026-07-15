# 🧠 一、PyTorch核心模块总览

| 模块                    | 作用               |
| --------------------- | ---------------- |
| `torch`               | 基础张量计算（类似 NumPy） |
| `torch.nn`            | 神经网络结构           |
| `torch.optim`         | 优化器              |
| `torch.utils.data`    | 数据加载             |
| `torch.autograd`      | 自动求导             |
| `torch.nn.functional` | 无状态函数层           |
| `torch.cuda`          | GPU 加速           |


# 📦 二、重点模块详解

## 1️⃣ `torch`（基础核心）

```python
import torch

x = torch.tensor([1,2,3])
x = x.to("cuda")  # 上GPU
```

功能：

* 张量（Tensor）
* 数学运算
* GPU/CPU切换

## 2️⃣ `torch.nn` 神经网络结构

👉 搭模型的核心模块

```python
import torch.nn as nn

model = nn.Linear(10, 5)
```

常见组件：
* `nn.Linear`（全连接层）
* `nn.Conv2d`（卷积）
* `nn.Embedding`
* `nn.Dropout`
* `nn.ReLU`

👉 还有一个关键类：

```python
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
```

## 3️⃣ `torch.nn.functional`（函数版层）

👉 和 `nn` 类似，但没有参数

```python
import torch.nn.functional as F
x = 1
x = F.relu(x)
```
relu 函数把负数变成 0，正数保持不变
区别：

| 写法          | 是否带参数 |
| ----------- | ----- |
| `nn.ReLU()` | ✅     |
| `F.relu()`  | ❌     |


## 4️⃣ `torch.optim`（优化器）

👉 负责训练时更新参数

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

常用：
| 优化器   | 特点    | 适用场景                |
| ----- | ----- | ------------------- |
| SGD   | 简单、稳定 | 传统CV、需要泛化           |
| Adam  | 快速、好用 | 大多数任务               |
| AdamW | 正则更合理 | ✅ Transformer / GPT |

🧠 一个直觉类比
SGD 👉 “每一步都凭感觉走”
Adam 👉 “会记住之前经验，还会调整步子大小”
AdamW 👉 “在 Adam 基础上，还控制‘体重’（防过拟合）”

## 5️⃣ `torch.utils.data`（数据加载）

👉 处理数据集 + 批处理

```python
from torch.utils.data import Dataset, DataLoader
```

核心：

* `Dataset` → 定义数据
* `DataLoader` → 批量读取 + shuffle

---

## 6️⃣ `torch.autograd`（自动求导）

👉 PyTorch 的核心黑科技

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x * x
y.backward()
```

👉 自动计算梯度

总结一句话
“记录计算过程，然后自动帮你反向求导”
关键点：
Transformer 里梯度是怎么在 Attention 中传播的（很重要）

梯度 = 函数变化最快的方向 + 变化的速度
假设你在爬山：
当前位置：山坡某一点
梯度告诉你：
往哪走最陡（方向）
坡有多陡（大小）

## 7️⃣ `torch.cuda`（GPU）

```python
import torch
torch.cuda.is_available()
```

👉 控制 GPU：

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

# 🧩 三、你做 GPT / Transformer 必用模块

结合你现在在做的内容👇

| 功能        | 模块                      |
| --------- | ----------------------- |
| embedding | `nn.Embedding`          |
| dropout   | `nn.Dropout`            |
| attention | `nn.MultiheadAttention` |
| loss      | `nn.CrossEntropyLoss`   |
| 优化器       | `optim.AdamW`           |

---

# 🚀 四、最小训练流程（串起来）

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

x = torch.randn(4, 10)
y = torch.randn(4, 2)

# forward
pred = model(x)

# loss
loss = loss_fn(pred, y)

# backward
loss.backward()

# update
optimizer.step()
optimizer.zero_grad()
```

---

# ✅ 总结一句话

👉 PyTorch 模块可以理解为：

> torch 负责算，nn 负责搭模型，optim 负责训练，data 负责喂数据
