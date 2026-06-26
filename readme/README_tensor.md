# 🧠 一句话理解 Tensor （张量）

👉 Tensor = **多维数组（加强版的数组）**
   Tensor（张量）**是 **PyTorch** 里最基础、最核心的数据结构。
   张量 = 带有形状（Shape）的多维数组
   Tensor 就是 PyTorch 的“多维数组”。支持GPU、支持自动求导、支持深度学习运算
---

# 📦 从简单到复杂理解

你可以这样类比👇

| 数据类型   | 类比            |
| ------ | ------------- |
| 标量     | 一个数字（0维）      |
| 向量     | 一维数组（1维）      |
| 矩阵     | 二维数组（2维）      |
| Tensor | **任意维数组（n维）** |

---

# 🔍 举例说明

```python
import torch
```

---

## 0️⃣ 标量（0维）

```python
x = torch.tensor(5)
```

👉 就一个数

---

## 1️⃣ 向量（1维）

```python
x = torch.tensor([1, 2, 3])
```

👉 一排数据

---

## 2️⃣ 矩阵（2维）

```python
x = torch.tensor([
    [1, 2],
    [3, 4]
])
```

👉 表格

---

## 3️⃣ 高维 Tensor

```python
x = torch.randn(2, 3, 4)
```

torch.randn(2, 3, 4) 生成一个 3维 Tensor（3D数组）
2 个 “3×4 的矩阵” 叠在一起
[
  [   # 第 1 个块
    [a, b, c, d],
    [e, f, g, h],
    [i, j, k, l]
  ],

  [   # 第 2 个块
    [m, n, o, p],
    [q, r, s, t],
    [u, v, w, x]
  ]
]

---

# 📐 Tensor 的关键属性

---

## 1️⃣ shape（形状）

```python
x.shape
```

👉 表示维度：

```text
(2, 3, 4)
```

---

## 2️⃣ dtype（数据类型）

```python
x.dtype
```

常见：

* `float32`（最常用）
* `int64`（Long）

---

## 3️⃣ device（设备）

```python
x.device
```

👉 在哪里：

* CPU
* GPU

---

# 🚀 Tensor 能做什么？

👉 一切计算都基于 Tensor

---

## ✔️ 数学运算

```python
x + y
x * y
```

---

## ✔️ 矩阵乘法

```python
x @ y
```

---

## ✔️ 自动求导（关键）

```python
x = torch.tensor(2.0, requires_grad=True)
```

👉 和 `autograd` 结合

---

# 🧠 在深度学习中的角色

---

## 📌 数据

```text
输入数据 → Tensor
```

---

## 📌 参数

```text
权重（weight） → Tensor
```

---

## 📌 中间结果

```text
hidden states → Tensor
```

---

👉 一切都是 Tensor

---

# 🔥 一个完整流转例子

```python
x = torch.randn(1, 10)   # 输入（Tensor）
w = torch.randn(10, 5)   # 权重（Tensor）

y = x @ w                # 计算（Tensor）
```

---

# ⚠️ 常见坑

---

## ❌ 1. 类型错误

```python
x = torch.tensor([1,2,3])  # 默认 Long
```

👉 不能做：

* GELU
* ReLU（有时会报错）

✔ 正确：

```python
x = x.float()
```

---

## ❌ 2. shape 不匹配

```python
x @ y
```

👉 维度必须对齐

---

## ❌ 3. CPU/GPU 混用

```python
x.cuda()
y.cpu()
x + y  # ❌ 报错
```

---

# 🆚 和 NumPy 的区别

| 特性    | Tensor | NumPy |
| ----- | ------ | ----- |
| GPU支持 | ✅      | ❌     |
| 自动求导  | ✅      | ❌     |
| 深度学习  | ✅      | ❌     |

---

# 🧠 一个直觉类比

👉 Tensor 就像：

**“神经网络世界里的数据容器 + 计算单位”**

---

# 🎯 总结

👉 Tensor 本质就是：

**“可以在 CPU/GPU 上进行高效计算的多维数组”**

---

# 🧩 程序员视角

```text
Tensor = 数据 + 形状 + 类型 + 设备 +（可选）梯度能力
```

