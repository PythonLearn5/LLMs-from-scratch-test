# 🧠 一句话理解 nn.Sequential

`nn.Sequential` 是 **PyTorch** 里一个非常实用的模块，用来**按顺序把多层网络“串起来”**。

# 📦 它解决什么问题？

通常你写网络要这样：

```
x = layer1(x)
x = layer2(x)
x = layer3(x)
```

👉 有点啰嗦

---

# 🚀 用 `nn.Sequential` 后

```
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)
```

👉 等价于：

```
x = Linear1(x)
x = ReLU(x)
x = Linear2(x)
```

---

# 🔍 执行过程

```text
输入 x
  ↓
Linear(10→20)
  ↓
ReLU
  ↓
Linear(20→5)
  ↓
输出
```

👉 自动按顺序调用

---

# ⚙️ 使用示例

```
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

x = torch.randn(1, 4)
y = model(x)

print(y.shape)
```

---

# 🧩 本质原理

👉 `Sequential` 内部其实就是：

```
for layer in layers:
    x = layer(x)
```

---

# 🆚 和自定义 `nn.Module` 的区别

## ✔️ `Sequential`（简单场景）

👉 适合：

* 纯“顺序结构”
* 一层接一层

## ✔️ `nn.Module`（复杂场景）

```
class MyModel(nn.Module):
    def forward(self, x):
        x1 = layer1(x)
        x2 = layer2(x)
        return x1 + x2
```

👉 支持：

* 分支
* 跳连接（residual）
* 多输入输出

# 🚨 限制（很重要）

`nn.Sequential` **不能做这些：**

❌ 分支结构
❌ 跳跃连接（ResNet / Transformer）
❌ 多输入


# 🧠 在 Transformer / GPT 中

👉 基本不用 `Sequential` 做整体模型

但会用在：

### ✔️ FFN（前馈网络）

```
self.ffn = nn.Sequential(
    nn.Linear(dim, dim * 4),
    nn.GELU(),
    nn.Linear(dim * 4, dim)
)
```

👉 这是标准写法


# 🧩 进阶写法（带名字）

```
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(10, 20)),
    ("relu", nn.ReLU()),
    ("fc2", nn.Linear(20, 5))
]))
```

👉 可以按名字访问


# 🎯 总结

👉 `nn.Sequential` 本质就是：

**“把一堆层按顺序打包，自动执行 forward”**


# 🧠 程序员视角

```text
Sequential = pipeline（流水线）
```


