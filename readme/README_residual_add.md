# Residual Add（残差连接）

这是 Transformer 中最重要的设计之一。Residual Add（残差连接） 可以说是 GPT-2 能训练几十层 Transformer 的关键技术之一。如果没有它，GPT-2 基本无法训练。

先看 GPT-2 一个完整的 Transformer Block（Pre-LN 结构）：

```text
输入 x
   │
   ▼
LayerNorm
   │
   ▼
Multi-Head Self-Attention
   │
   ▼
Residual Add (+ 原输入)
   │
   ▼
LayerNorm
   │
   ▼
MLP (Feed Forward)
   │
   ▼
Residual Add (+ Attention输出)
   │
   ▼
输出
```

对应公式：

第一部分（Attention）

```text
a = x + Attention(LN(x))
```

第二部分（MLP）

```text
y = a + MLP(LN(a))
```

这里的 + 就是 Residual Add。

---

# 一、Residual Add 是什么？

实际上就是：

```python
output = input + sublayer(input)
```

例如：

```python
x = torch.tensor([1,2,3])

attention_out = torch.tensor([0.1,0.5,-0.2])

output = x + attention_out
```

得到

```text
[1.1, 2.5, 2.8]
```

没有任何复杂计算，就是逐元素相加。

例如

```
输入 [1 2 3]
Attention 输出[0.1 0.5-0.2]
Residual Add
= [1.1 2.5 2.8]
```

---

# 二、为什么不直接输出 Attention？

假设没有 Residual：

```text
output = Attention(x)
```

那么

```
输入

苹果 很 红

↓

Attention

↓

新的表示
```

Attention 会把原来的信息全部覆盖。

如果 Attention 学得不好：

```
苹果
↓↓↓

垃圾输出
```

整层的信息就毁掉了。

而 Residual：

```
苹果

↓

Attention 得到修正

↓

苹果
+
修正

↓

更好的苹果表示
```

注意：

Attention 不再负责重新生成全部信息。

它只负责：

> 修改（Refine）输入。

因此很多论文都会说：

> Transformer 学的是 Residual Function（残差函数）。

也就是：

```
输出 = 输入 + 修正量
```

而不是

```
输出 = 全新的东西
```

---

# 三、Residual 学的是"增量"

例如：

输入 embedding

```
猫

↓

[0.8 0.2 0.5]
```

Attention 学到：

```
因为上下文是：

那只 猫 睡觉

应该增加一点"动物"

减少一点"颜色"

```

于是 Attention 输出：

```
[+0.02 -0.01 +0.05]
```

Residual：

```
原来：

[0.80 0.20 0.50]

+

修正：

[0.02 -0.01 0.05]

=

[0.82 0.19 0.55]
```

是不是比：

```
直接输出

[0.82 0.19 0.55]
```

更容易学习？

答案是：

容易得多。

---

# 四、为什么更容易训练？

假设真实函数：

```
F(x)=x
```

也就是说：

其实这一层什么都不用干。

没有 Residual：

网络必须学习：

```
Attention(x)=x
```

必须精确复制输入。

这是很困难的。

有 Residual：

```
output=x+Attention(x)
```

网络只需要学习：

```
Attention(x)=0
```

即可。

学习

```
0
```

远远比学习

```
复制整个 x
```

简单。

---

# 五、梯度传播为什么更稳定？

这是 Residual 最重要的作用。

没有 Residual：

```
x

↓

Layer1

↓

Layer2

↓

Layer3

↓

Loss
```

梯度：

```
Loss

↓

Layer3

↓

Layer2

↓

Layer1
```

梯度一路乘：

```
0.9

×

0.8

×

0.7

×

...
```

越来越小：

```
0.9×0.8×0.7≈0.504
```

几十层以后：

```
≈0
```

梯度消失。

---

加入 Residual：

```
x
 │
 │──────────────┐
 ▼              │
Attention       │
 ▼              │
 +──────────────┘
 │
 ▼
输出
```

梯度现在有两条路：

```
Loss
 ↓
 Add
 ↙   ↘

Attention   Identity
```

Identity（恒等映射）这条路径不会引入额外变换，因此梯度可以更直接地传回前面的层，缓解梯度消失问题。

从数学上看：

```
y = x + F(x)
```

求导：

```
dy/dx = I + dF/dx
```

这里的 `I` 是恒等映射的导数（单位矩阵）。

即使：

```
dF/dx ≈ 0
```

仍然有：

```
dy/dx ≈ I
```

意味着梯度至少还有一条"直通"路径。

---

# 六、为什么 GPT2 每个子层都要 Residual？

GPT2 一个 Block 有两个子层：

```
Attention
```

和

```
MLP
```

它们都只负责：

> 在已有表示上做修正。

因此都是：

```
输入
↓

Attention

↓

Residual
```

再

```
输入
↓

MLP

↓

Residual
```

这样每个模块都学习"增量"而不是"重建全部表示"。

---

# 七、Residual 可以理解成"边保留边修改"

可以把它想象成修改文档：

原文：

```
今天 天气 很好
```

Attention 像一个编辑器：

它不是重新写一篇文章，而是提出修改意见：

```
很好

↓

非常好
```

Residual 就相当于：

```
原文
+
修改
=
最终版本
```

模型保留了已有信息，同时叠加新的上下文信息。

---

## 总结

Residual Add 在 GPT-2 Transformer Block 中有四个核心作用：

1. 保留原始信息：避免 Attention 或 MLP 的输出完全覆盖输入表示。
2. 学习增量（Residual Function）：子层只需学习对输入的修正，而不是重建整个表示，优化更容易。
3. 改善梯度传播：恒等映射提供了梯度直通路径，显著缓解深层网络中的梯度消失问题。
4. 支持深层网络训练：正是残差连接配合 LayerNorm 等设计，使得 GPT-2、GPT-3、Llama 等拥有数十甚至上百层的 Transformer 能够稳定训练。

一句话概括就是：Residual Add 的核心思想不是"替换"输入，而是在保留原有信息的基础上不断"修正"表示，使信息流和梯度流都更加稳定。
