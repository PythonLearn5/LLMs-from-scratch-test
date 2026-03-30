# 🧠 nn.Parameter 理解

👉 `nn.Parameter` = **会被模型自动训练（更新）的张量**

---

# 📦 它解决什么问题？

在 PyTorch 里：

```
self.xxx = 某个张量
```
👉 默认不会被当成“模型参数”
也就是说：
❌ 优化器不会更新它

---

# 🚀 `nn.Parameter` 的作用

```
import torch.nn as nn
import torch

w = nn.Parameter(torch.randn(3, 3))
```

👉 这时：

* `w` 会被自动加入 `model.parameters()`
* 会参与训练（反向传播 + 更新）

---

# 🔍 举个对比（非常重要）

## ❌ 普通 tensor

```
self.w = torch.randn(3, 3)
```

👉 结果：

* 不会被训练 ❌
* optimizer 看不到 ❌

---

## ✅ Parameter

```
self.w = nn.Parameter(torch.randn(3, 3))
```

👉 结果：

* 会被训练 ✅
* optimizer 能更新 ✅

---

# ⚙️ 内部本质

```
nn.Parameter = Tensor + requires_grad=True + 注册到模型
```

👉 等价于：

```
tensor.requires_grad = True
```

但多了一步：

👉 **自动注册到 Module**

---

# 🧩 在模型中的典型用法

```
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))
```

# 🔎 查看参数

```
for p in model.parameters():
    print(p.shape)
```

👉 能看到 `nn.Parameter`

---

# 🚀 在 Transformer / GPT 中的应用

## 1️⃣ 位置编码（可学习）

```
self.pos_embedding = nn.Parameter(torch.randn(max_len, dim))
```

---

## 2️⃣ LayerNorm 参数

```text
γ（scale） 和 β（bias）
```

👉 本质就是 `nn.Parameter`

---

## 3️⃣ 自定义权重

比如你自己实现：

* attention
* gating
* scaling

---

# ⚠️ 常见坑

---

## ❗ 1. 忘记用 Parameter

```
self.w = torch.randn(...)  # ❌
```

👉 训练完全没效果（很多人踩坑）

---

## ❗ 2. requires_grad=False

```
nn.Parameter(..., requires_grad=False)
```

👉 不会更新

---

## ❗ 3. Parameter 不能随便替换

```
self.w = some_tensor  # ❌ 会丢失注册
```

👉 正确：

```
self.w.data = some_tensor
```

---

# 🧠 和 Buffer 的区别（进阶）

| 类型                      | 是否训练  |
| ----------------------- | ----- |
| Parameter               | ✅ 会更新 |
| buffer（register_buffer） | ❌ 不更新 |

---

# 🎯 总结

👉 `nn.Parameter` 本质就是：

**“告诉 PyTorch：这个张量是模型的一部分，需要被训练”**

---

# 🧩 程序员视角总结

```text
Tensor        → 普通数据
Parameter     → 可训练参数（权重）
```
