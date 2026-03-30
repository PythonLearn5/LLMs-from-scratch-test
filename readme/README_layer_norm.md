
# 🧠 LayerNorm（层归一化）

👉 LayerNorm = **把一层内部的数据“拉到同一尺度”，让训练更稳定**

# 📐 它在做什么？

对一个向量（比如一个 token 的 embedding）：

1. 计算平均值（mean）
2. 计算方差（variance）
3. 做标准化

然后再加上可学习参数：

[
y = \gamma \hat{x} + \beta
]

👉 让模型还能“调回来”

# 🔍 举个直观例子

假设某个 token 的 embedding：

```text
[10, 20, 30]
```

LayerNorm 后：

```text
[-1.22, 0, 1.22]（大致）
```

👉 特点：

* 均值 = 0
* 方差 = 1
* 数值范围稳定

# 📦 在 PyTorch 中怎么用？

```python
import torch.nn as nn

ln = nn.LayerNorm(normalized_shape=768)

x = ln(x)
```

👉 `768` 通常是 embedding 维度（比如 GPT）

# 🧠 为什么需要 LayerNorm？

深度网络会遇到问题：

👉 每一层输出分布都在变（训练不稳定）

---

## 🚨 没有 LayerNorm 会怎样？

* 梯度不稳定
* 收敛慢
* 甚至训练崩掉

---

## ✅ 有了 LayerNorm

* 数值稳定
* 梯度更平滑
* 更容易训练深层网络

---

# 🆚 和 BatchNorm 的区别（重点）

| 对比              | LayerNorm  | BatchNorm |
| --------------- | ---------- | --------- |
| 归一化维度           | **每个样本内部** | batch 之间  |
| 是否依赖 batch size | ❌ 不依赖      | ✅ 依赖      |
| NLP 是否适合        | ✅ 非常适合     | ❌ 不适合     |

---

👉 Transformer 必须用 LayerNorm 的原因：

* NLP 任务 batch size 不稳定
* 序列长度变化
* 需要逐 token 处理

---

# 🚀 在 Transformer 里的位置

典型结构（GPT）：

```text
x → LayerNorm → Attention → Add
x → LayerNorm → MLP → Add
```

👉 叫做：

**Pre-LN 结构（现代主流）**

---

# 🎯 核心作用总结

LayerNorm 做了三件事：

1. **防止数值爆炸 / 消失**
2. **稳定训练**
3. **加快收敛**

---

# 🧠 再给你一个直觉类比

👉 LayerNorm 就像：

**把每一层的输出“统一到同一个标准”**

就像考试：

* 有的卷子满分100
* 有的满分150

👉 LayerNorm = 全部统一成“标准分”

---

# ✅ 总结一句话

👉 LayerNorm 本质就是：

**“让每一层的数据分布保持稳定，从而让深度网络更容易训练”**
