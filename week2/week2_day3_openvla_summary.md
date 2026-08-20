# Week 2 Day 3：OpenVLA 学习总结

## 1. 今日学习目标

理解 OpenVLA 如何把视觉、语言和机器人动作连接起来，重点掌握：

- Vision Encoder
- Projector
- LLM
- Action Token
- Action 离散化与反离散化
- 1%–99% percentile clipping
- OpenVLA 中 Action 的基本表示方式

---

## 2. OpenVLA 的整体结构

OpenVLA 的主要数据流可以概括为：

```text
Image
↓
Vision Encoder
↓
Visual Features
↓
Projector
↓
Visual Embeddings
        +
Language Tokens
        ↓
       LLM
        ↓
Action Tokens
        ↓
反离散化
        ↓
Robot Action
```

核心思想是：

> 把机器人动作也表示成 token，使 LLM 能够像预测文本 token 一样预测机器人 Action。

---

## 3. Vision Encoder

Vision Encoder 负责从机器人相机图像中提取视觉特征。

OpenVLA 使用视觉编码器处理图像，得到能够描述场景、物体和空间信息的视觉表示。

它解决的是：

```text
原始 Image
↓
提取视觉语义
↓
Visual Features
```

这些视觉特征还不能直接交给 LLM，因此后面需要 Projector。

---

## 4. Projector

Projector 位于：

```text
Vision Encoder
↓
Projector
↓
LLM
```

它的主要作用是：

> 把 Vision Encoder 输出的视觉特征转换到 LLM 能够接收的表示空间。

因此 Projector 不是直接完成图像和语言的语义匹配，而是负责连接视觉编码器和 LLM。

可以简单理解为：

```text
Visual Feature
↓
Projector
↓
LLM 可以理解的 Visual Embedding
```

---

## 5. LLM

经过 Projector 后，视觉信息可以和 Language Tokens 一起输入 LLM。

因此 LLM 同时获得：

```text
Visual Embeddings
+
Language Tokens
```

模型根据当前图像和语言指令预测后续的 Action Tokens。

整体关系为：

```text
Image + Language
↓
LLM
↓
Action Token
```

这里体现了 OpenVLA 的核心思想：

> 把 VLM 的视觉—语言理解能力继续扩展到机器人动作生成。

---

## 6. Action Token

机器人真实动作本身通常是连续值，例如：

```text
[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
```

但 LLM 处理的是离散 token。

因此 OpenVLA 需要先把连续 Action 离散化：

```text
Continuous Action
↓
离散化
↓
Action Token
```

模型预测 Action Token 后，再进行反离散化：

```text
Action Token
↓
反离散化
↓
Continuous Robot Action
```

因此 OpenVLA 并不是直接输出任意连续浮点动作，而是：

> 先预测离散 Action Token，再恢复为连续控制量。

---

## 7. Action 离散化

连续 Action 的每个维度都会根据数据分布映射到离散区间。

例如：

```text
真实动作值：0.034
↓
对应某个离散 bin
↓
Action Token
```

模型训练的目标就是预测正确的 Action Token。

推理时再把预测得到的 token 映射回对应的连续动作值。

---

## 8. 1%–99% Percentile Clipping

OpenVLA 在 Action 离散化前不会简单使用全部极值作为范围，而是使用约 1%–99% percentile 的范围进行处理。

这样做的原因是：

> 避免极少数异常大的 Action 值把整个离散化范围拉得过宽。

例如，大多数动作集中在较小范围：

```text
-0.1 ~ 0.1
```

但存在极少数：

```text
-1.5
1.8
```

如果直接用最大值和最小值划分离散区间，大量正常动作会被压缩到很少的 bin 中。

使用 percentile clipping 后，可以让主要 Action 数据获得更合理的离散精度。

---

## 9. OpenVLA 的 Action 是什么

当天形成的一个重要理解是：

> OpenVLA 学出来的动作本质上仍然是 Action Token，经过反离散化后恢复为对应的连续 Action。

在机器人控制任务中，常见表示包括末端执行器的相对变化，例如：

```text
[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
```

因此模型预测的是动作表示，具体如何由机器人底层控制系统转成关节执行，不属于 OpenVLA 的主要学习部分。

---

## 10. OpenVLA 的核心认识

OpenVLA 可以压缩成下面这条链：

```text
Image
↓
Vision Encoder
↓
Projector
↓
LLM ← Language
↓
Action Token
↓
反离散化
↓
Continuous Action
```

其中：

- Vision Encoder：负责看图。
- Projector：把视觉特征转换成 LLM 可接收的表示。
- LLM：融合视觉和语言信息并预测 Action Token。
- Action Token：连续机器人动作的离散表示。
- 反离散化：把预测 token 恢复为连续动作。
- Percentile clipping：减弱极端 Action 值对离散化的影响。

---

## 11. Day 3 最终应掌握

完成 Day 3 后，应能够解释：

1. OpenVLA 的输入和输出分别是什么。
2. Vision Encoder、Projector、LLM 各自负责什么。
3. 为什么连续机器人 Action 要转换成 Action Token。
4. Action Token 如何恢复成连续动作。
5. 为什么 Action 离散化时使用 1%–99% percentile clipping。
6. OpenVLA 的完整链路：

```text
Vision Encoder
→ Projector
→ LLM
→ Action Token
→ Action Decode
```

