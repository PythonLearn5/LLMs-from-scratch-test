# 模型的配置参数

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词表大小（GPT2 BPE tokenizer 的词汇量）
    "context_length": 1024, # 最大上下文长度（最大 token 数）
    "emb_dim": 768,         # token embedding 维度
    "n_heads": 12,          # 多头注意力的头数量
    "n_layers": 12,         # Transformer block 层数
    "drop_rate": 0.1,       # Dropout 比例（训练时随机丢弃神经元）
    "qkv_bias": False       # Q K V 线性层是否使用 bias
}

👉 GPT-2 (124M) 本质是：

* 12 层 Transformer
* 每层 12 个 attention heads
* 每个 token 用 768 维向量表示


## 1️⃣ `vocab_size = 50257`

### ✅ 作用

👉 词表大小（token 种类数）

* GPT-2 使用 BPE（子词分词）
* 一共 50257 个 token

---

### 🧠 直觉

```text
"hello" → 可能是一个 token  
"unbelievable" → 可能拆成多个 token
```

---

### 📌 影响

* 决定 embedding 矩阵大小：

```text
[50257 × 768]
```

* 越大 → 表达更细，但模型更大

---

## 2️⃣ `context_length = 1024`

### ✅ 作用

👉 一次最多处理多少个 token（上下文窗口）

---

### 🧠 直觉

```text
最多记住 1024 个词（超过就“忘前面”）
```

---

### 📌 影响

* Attention 计算复杂度：

```text
O(n²)
```

👉 1024 已经很吃算力了

---

## 3️⃣ `emb_dim = 768`

### ✅ 作用

👉 每个 token 表示成一个 768 维向量

---

### 🧠 直觉

```text
"猫" → [0.12, -0.8, 0.33, ... 共768维]
```

---

### 📌 影响

* 维度越大 → 表达能力越强
* 但计算量也更大

---

## 4️⃣ `n_heads = 12`

### ✅ 作用

👉 多头注意力的头数

---

### 🧠 关键关系（非常重要）：

```text
每个 head 维度 = 768 / 12 = 64
```

---

### 📌 影响

* 多头 = 多角度理解
* 但 head 太多：

  * 每个 head 变小（信息变少）

---

## 5️⃣ `n_layers = 12`

### ✅ 作用

👉 Transformer 堆叠层数

---

### 🧠 直觉

* 每一层都在“重新理解一句话”

```text
Layer1：基础语义  
Layer6：句法关系  
Layer12：高级语义
```

---

### 📌 影响

* 层数越多 → 理解更深
* 但：

  * 更慢
  * 更难训练

---

## 6️⃣ `drop_rate = 0.1`

### ✅ 作用

👉 Dropout 概率（训练用）

---

### 🧠 直觉

```text
训练时随机“关掉”10%神经元
```

---

### 📌 影响

* 防止过拟合
* 推理时不生效（自动关闭）

---

## 7️⃣ `qkv_bias = False`

### ✅ 作用

👉 Q / K / V 线性层是否加 bias

---

### 🧠 解释

线性层通常是：

```text
y = Wx + b
```

这里决定：

```text
要不要 b（偏置项）
```

---

### 📌 GPT-2 为什么设为 False？

* 简化模型
* LayerNorm 已经提供偏移能力
* 实践中效果足够好

---

# 三、关键关系总结（很重要）

### 🔥 1. 维度拆分

```text
emb_dim = n_heads × head_dim
768 = 12 × 64
```

---

### 🔥 2. 参数量来源（为什么叫 124M）

主要来自：

* Embedding
* Attention（QKV）
* FFN（前馈网络）

👉 总计约 1.24 亿参数

---

# 四、一句话总结

👉 这组配置定义了 GPT-2 的“体型 + 大脑结构”：

* vocab_size：认识多少词
* context_length：能记多长
* emb_dim：每个词多聪明
* n_heads：看问题的角度数
* n_layers：思考深度
* drop_rate：防过拟合
* qkv_bias：是否加偏置

---
