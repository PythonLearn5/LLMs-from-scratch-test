# 🧠 一句话理解

`nn.Linear` = **对输入做一次线性变换（矩阵乘法 + 偏置）**
`nn.Linear` 是 **PyTorch** 里最基础、最常用的神经网络层之一，也叫**全连接层（Fully Connected Layer）**。

---

# 📐 数学本质

y = xW^T + b

👉 含义：

* (x)：输入
* (W)：权重（模型要学的）
* (b)：偏置（模型要学的）
* (y)：输出

---

# 📦 基本用法

```python
import torch.nn as nn

layer = nn.Linear(in_features=4, out_features=2)
```

👉 表示：

* 输入维度：4
* 输出维度：2

---

# 🔍 输入输出结构

假设输入：

```python
x.shape = (batch_size, 4)
```

经过：

```python
y = layer(x)
```

输出：

```python
y.shape = (batch_size, 2)
```

---

# 📊 举个具体例子

```python
import torch
import torch.nn as nn

layer = nn.Linear(3, 2)

x = torch.tensor([[1.0, 2.0, 3.0]])
y = layer(x)

print(y)
```

👉 内部做的其实是：

```text
[1,2,3] × W + b → [y1, y2]
```

---

# 🧩 参数长什么样？

```python
layer.weight.shape  # (2, 3)
layer.bias.shape    # (2,)
```

👉 解释：

* 2 行 → 输出维度
* 3 列 → 输入维度

---

# 🧠 直观理解

👉 `nn.Linear(3, 2)` 做的事情：

```text
3个输入 → 变成 → 2个输出
```

就像：

* 输入：身高、体重、年龄（3维）
* 输出：两个评分（2维）

---

# 🚀 在神经网络中的作用

---

## 1️⃣ 特征变换

```text
输入特征 → 新特征空间
```

---

## 2️⃣ 改变维度（非常重要）

```text
(…, 768) → (…, 3072)
```

👉 Transformer 里常见

---

## 3️⃣ 最终输出层

```text
hidden → 分类结果
```

---

# 🧠 在 GPT / Transformer 中

---

## 📌 FFN（前馈网络）

```python
nn.Linear(dim, dim * 4)
nn.GELU()
nn.Linear(dim * 4, dim)
```

👉 核心结构

---

## 📌 Attention 里的投影

```text
Q = Linear(x)
K = Linear(x)
V = Linear(x)
```

👉 全是 Linear

---

# ⚠️ 常见坑

---

## ❌ 1. 输入维度不匹配

```python
nn.Linear(4, 2)
x.shape = (batch, 5)  # ❌ 报错
```

---

## ❌ 2. 忘记 batch 维

```python
x.shape = (4,)  # 有时会出问题
```

---

## ❌ 3. 类型错误

👉 必须是 float：

```python
x = x.float()
```

---

# 🆚 和数学里的区别

👉 数学：

```text
y = Wx
```

👉 PyTorch：

```text
y = xW^T
```

👉 只是存储方式不同，本质一样

---

# 🎯 总结

👉 `nn.Linear` 本质就是：

**“用矩阵乘法把输入映射到新的空间”**

---

# 🧩 程序员视角

```text
Linear = 可训练的矩阵变换
```

---
