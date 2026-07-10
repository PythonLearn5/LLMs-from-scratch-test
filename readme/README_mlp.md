MLP（Multi-Layer Perceptron，多层感知机） 是一种最基础的神经网络结构，本质上就是多个 `nn.Linear`（全连接层）叠加，再加上激活函数。

简单理解：

> MLP = Linear + 激活函数 + Linear + 激活函数 + ...

它负责把输入特征进行非线性变换，学习更复杂的模式。

---

## 1. 最简单的 MLP

例如：

```python
import torch
import torch.nn as nn

mlp = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 2)
)
```

结构：

```
输入3维向量

[ x1 x2  x3 ]
      |
      v
Linear(3 → 4)
      |
      v
ReLU
      |
      v
Linear(4 → 2)
      |
      v
输出2维向量
```

---

## 2. 每一层做什么？

假设输入：

```
x = [2,3,4]
```

### 第一层 Linear

```python
nn.Linear(3,4)
```

内部：

[h=xW+b]

得到：

```
h = [1.2,-0.5,3.1,2.7]
```

现在从 3 个特征变成 4 个新特征。

---

### 激活函数 ReLU

```python
ReLU(x)=max(0,x)
```

变成：

```
[1.2,0,3.1,2.7]
```

作用：

加入非线性能力。

如果没有 ReLU：

```
Linear + Linear
```

实际上还是一个 Linear：

[
xW_1W_2
]

无法学习复杂关系。

---

### 第二层 Linear

```python
nn.Linear(4,2)
```

再映射：

```
[1.2,0,3.1,2.7]
↓
[0.8,5.6]
```

最终输出。

---

# 3. GPT 里的 MLP 是什么？

Transformer 每个 Block 里面都有：

```
Transformer Block

        |
        |
   Self Attention
        |
        |
      Add&Norm
        |
        |
       MLP
        |
        |
      Add&Norm
```

GPT 中的 MLP 又叫：

* Feed Forward Network（FFN）
* 前馈网络

典型结构：

```
输入768维
   |
   v
Linear 768 → 3072
   |
   v
GELU激活
   |
   v
Linear 3072 → 768
   |
   v
输出 768维
```

---

## 为什么先扩大维度？

例如 GPT-2：

隐藏维度：

```
768
```

MLP：

```
768 → 3072 → 768
```

原因：

升维：

```
768个特征
      |
      |
      v
3072个特征
```

让模型有更多空间组合信息。

例如：

输入：

```
猫
坐
垫子
```

经过 MLP 后可能产生：

```
动物相关特征
动作相关特征
位置相关特征
语义关系特征
...
```

然后再压回：

```
768维
```

---

# 4. MLP 和 Attention 的区别

这是 Transformer 理解重点：

|      | Attention | MLP        |
| ---- | --------- | ---------- |
| 作用   | 看其他 Token | 处理当前 Token |
| 是否交流 | ✅ 会和其他词交互 | ❌ 不交流      |
| 输入   | 所有 token  | 单个 token   |
| 学习方式 | QKV权重     | Linear权重   |
| 作用   | 信息选择      | 特征加工       |

例如：

一句话：

```
小明 去 银行 取钱
```

Attention：

```
银行
 ↔
取钱

发现银行是金融机构
```

MLP：

```
把银行这个token的信息
进一步加工
形成更丰富特征
```

---

# 5. 一个 Transformer Block 可以理解成

```
输入 Token Embedding
        |
        |
        v
+----------------+
| Self Attention |
|  找相关信息     |
+----------------+
        |
        v
+----------------+
| MLP            |
|  加工信息       |
+----------------+
        |
        v

新的 Token 表示
```

---

一句话总结：

> MLP 就是一组 Linear 层组成的小型神经网络，在 Transformer 中负责对 Attention 得到的信息进行深度加工和特征提取。Attention 负责“找谁重要”，MLP 负责“理解和转换这些信息”。
