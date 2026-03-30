# 🧠 一句话理解

`nn.Module` 是 **PyTorch** 里最核心的类，几乎**所有模型、层、组件都是基于它构建的**。

# 📦 它解决什么问题？

在深度学习中，你需要：

* 定义模型结构
* 管理参数
* 做前向计算（forward）
* 支持训练（反向传播）

👉 `nn.Module` 把这些全部封装好了

# 🚀 最基本用法

```
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
```

👉 使用：

```
model = MyModel()
y = model(x)  # 自动调用 forward
```

# 🔍 核心组成（必须掌握）

## 1️⃣ `__init__`（定义结构）

```
self.fc = nn.Linear(10, 5)
```

👉 在这里定义：

* 层（Layer）
* 参数（Parameter）

## 2️⃣ `forward`（定义计算）

```
def forward(self, x):
    return self.fc(x)
```

👉 定义数据如何流动

## 3️⃣ 自动调用机制

```
model(x)
```

👉 实际调用：

```
model.__call__(x) → forward(x)
```

# 🧩 它帮你自动做的事（重点）

## ✅ 1. 管理参数

```
model.parameters()
```

👉 自动收集：

* `nn.Linear`
* `nn.Parameter`

## ✅ 2. 支持 GPU

```
model.to("cuda")
```
👉 所有参数一起移动

## ✅ 3. 支持训练 / 推理模式

```
model.train()
model.eval()
```

👉 控制：
* Dropout
* BatchNorm

## ✅ 4. 支持保存 / 加载

```
torch.save(model.state_dict(), "model.pth")
```

# 🔍 举个稍微复杂一点的例子

```
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

    def forward(self, x):
        return self.net(x)
```

👉 `Sequential` 也是 `Module`

# 🆚 和普通类的区别

| 普通 Python 类 | nn.Module |
| ----------- | --------- |
| 只是代码结构      | ✅ 带深度学习功能 |
| 不管理参数       | ✅ 自动管理    |
| 不支持训练       | ✅ 支持      |

# 🚀 在 GPT / Transformer 中

你现在学的模型其实是：

```text
GPTModel (nn.Module)
 ├── Embedding (nn.Module)
 ├── TransformerBlock (nn.Module)
 │     ├── Attention
 │     ├── FFN
 │     └── LayerNorm
```
👉 **一切都是 Module 嵌套 Module**

# ⚠️ 常见错误

## ❌ 忘记继承

```
class MyModel:  # ❌
```

## ❌ 忘记 super()

```
super().__init__()  # 必须写
```

## ❌ forward 不写

👉 模型无法运行

# 🧠 一个非常重要的理解

👉 `nn.Module` 本质是一个“树结构”

```text
model
 ├── layer1
 ├── layer2
 └── submodule
      ├── ...
```
👉 PyTorch 会递归管理所有子模块

# 🎯 总结

👉 `nn.Module` 本质就是：

**“用来定义神经网络结构 + 管理参数 + 支持训练的一切基础”**

# 🧩 程序员视角

```text
nn.Module = 可训练对象（带生命周期 + 参数管理 + forward逻辑）
```
