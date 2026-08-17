# Week 2：RT-1 与 RT-2 知识点总结

## Day 1：RT-1

### 1. RT-1 的核心目标

RT-1 是一个基于 Vision 和 Language 的机器人控制策略。

```text
Image History + Language Instruction
                ↓
              RT-1
                ↓
             Action_t
```

本质仍然是：

```text
Observation → Policy → Action
```

只是 Observation 被扩展成了：

```text
Image History + Language
```

---

### 2. RT-1 的输入

RT-1 的主要输入包括：

* 最近一段 Image History
* Language Instruction

Image History 可以提供时间信息，例如判断机械臂：

* 正在靠近目标
* 正在远离目标
* 是否已经抓住物体

因此 RT-1 不只根据单张图像决策。

---

### 3. FiLM

Language Instruction 经过编码后，通过 FiLM 影响视觉特征提取。

可以简单理解为：

```text
Language
   ↓
告诉视觉模型
“当前任务应该重点关注什么”
```

例如：

```text
Instruction：
Pick up the red cup
```

模型应该更多关注：

* red cup
* gripper
* cup 与 gripper 的空间关系

因此 FiLM 的作用是让视觉特征具有 **task-conditioned** 特性。

---

### 4. EfficientNet

EfficientNet 负责从原始图像中提取视觉特征：

```text
Image
  ↓
EfficientNet
  ↓
Visual Features
```

结合 FiLM 后，得到与当前语言任务相关的视觉信息。

---

### 5. TokenLearner

EfficientNet 会产生大量视觉特征。

TokenLearner 的作用是：

```text
大量 Visual Features
        ↓
   TokenLearner
        ↓
少量重要 Visual Tokens
```

RT-1 中每帧最终压缩成少量 token，再将连续多帧的 token 输入 Transformer。

目的：

* 减少计算量
* 保留任务相关的重要视觉信息

---

### 6. Transformer

Transformer 是 RT-1 的核心 Policy。

它学习：

```text
Language + Visual History
           ↓
       Transformer
           ↓
当前 timestep 应该执行什么 Action
```

这个映射通过 Expert Demonstration 的监督学习得到。

训练数据本质上是：

```text
(Image History, Language) → Expert Action
```

因此 RT-1 仍然可以理解为一种 Behavioral Cloning：

```text
Observation
    ↓
  Policy
    ↓
 Action
```

---

### 7. RT-1 的 Action 表示

一个 timestep 的 action 可以表示成：

```text
[Δx, Δy, Δz,
 Δroll, Δpitch, Δyaw,
 gripper, ...]
```

其中很多维度原本是连续值。

例如：

```text
Δx = +0.02
```

RT-1 不直接回归这个连续值，而是将每个 action dimension 的范围划分成：

```text
256 bins
```

编号：

```text
0 ~ 255
```

---

### 8. Action 离散化

例如：

```text
Δx ∈ [min, max]
```

将这个范围均匀划分为 256 个区间：

```text
min                              max
 |--------------------------------|
 0  1  2  3 ...               255
```

假设专家动作：

```text
Δx = +0.02
```

落入第 179 个 bin：

```text
Δx = +0.02
      ↓
   bin 179
      ↓
Action Token = 179
```

因此：

```text
Continuous Action
        ↓
Discretization
        ↓
Action Token
```

---

### 9. Action 反离散化

模型预测：

```text
Action Token = 179
```

系统根据提前定义好的 bin 范围，将其恢复成近似连续动作：

```text
179
 ↓
对应第 179 个动作区间
 ↓
Δx ≈ +0.02
```

因此：

```text
Action Token
     ↓
De-discretization
     ↓
Continuous Action
     ↓
Robot Execution
```

需要注意：

> 模型学习的是“应该选择哪个 bin”。

而：

> “bin 179 对应多少实际动作”是提前定义好的。

---

### 10. RT-1 如何形成 Trajectory

RT-1 一次主要预测一个 timestep：

```text
Observation_t
     ↓
   RT-1
     ↓
 Action_t
```

执行后得到新的 Observation：

```text
Action_t
   ↓
Robot Execution
   ↓
Observation_(t+1)
   ↓
RT-1
   ↓
Action_(t+1)
```

不断重复：

```text
a_1 → a_2 → a_3 → ... → a_T
```

最终形成完整 trajectory。

因此：

> RT-1 的 trajectory 是一个 timestep 一个 timestep 闭环生成出来的，而不是一次预测完整 trajectory。

---

## Day 2：RT-2

### 1. RT-2 的核心目标

RT-2 的核心不是重新训练一个机器人专用模型，而是：

```text
Pretrained VLM
      +
Robot Demonstrations
      ↓
      RT-2
      ↓
      VLA
```

即：

```text
Vision-Language Model
        ↓
Vision-Language-Action Model
```

---

### 2. RT-1 与 RT-2 最大区别

RT-1：

```text
Robot Demonstrations
        ↓
训练 Robot Policy
        ↓
Action
```

RT-2：

```text
已经预训练好的 VLM
        ↓
加入 Robot Demonstrations
        ↓
继续监督微调
        ↓
学会输出 Action
```

因此 RT-2 的核心优势是：

> 可以利用 VLM 已经学到的大量 Vision、Language 和 Web 世界知识。

---

### 3. 为什么 VLM 可以输出 Action

普通 VLM 原本执行：

```text
Image + Language
       ↓
      VLM
       ↓
Text Tokens
```

例如：

```text
The → cup → is → red
```

本质上是在进行：

```text
Next-Token Prediction
```

即：

> 根据已有输入和前面的 token，预测下一个 token。

---

### 4. RT-2 将 Action Token 化

机器人原始动作：

```text
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
```

先进行离散化：

```text
Δx → 145
Δy → 112
Δz → 128
...
```

得到：

```text
[145, 112, 128, ...]
```

于是机器人 Action 也变成了一串 token。

---

### 5. Action Token 为什么可以使用 Next-Token Prediction

对于 VLM：

```text
"apple"
```

是一个 token。

对于 RT-2：

```text
action_token_145
```

同样也是一个离散 token。

虽然二者含义完全不同，但从模型输出形式上都是：

```text
预测一个离散 token
```

因此原来的：

```text
The → red → cup
```

可以扩展成：

```text
145 → 112 → 128 → ...
```

使用相同的 next-token prediction 方式。

---

### 6. RT-2 的监督训练

Robot Demonstration 提供：

```text
Robot Image
+
Language Instruction
+
Expert Action
```

Expert Action 首先被转换成 Action Tokens：

```text
Expert Action
     ↓
Discretization
     ↓
GT Action Tokens
```

然后训练：

```text
Image + Instruction
        ↓
Pretrained VLM
        ↓
Predicted Action Tokens
        ↓
与 GT Action Tokens 比较
        ↓
监督微调
```

因此：

> VLM 并不是天然就会控制机器人，而是利用 Robot Demonstration 学习 Vision + Language → Action Token 的映射。

---

### 7. Next-Token 不等于 Next-Timestep

这是 RT-2 最重要的概念之一。

例如当前 timestep：

```text
action_t =
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
```

对应：

```text
145 → 112 → 128 → ...
```

这里的：

```text
next-token
```

是在依次预测 **同一个 timestep 内不同 action dimension 对应的 token**。

不是：

```text
action_t
→ action_(t+1)
→ action_(t+2)
```

因此：

```text
Next Token ≠ Next Timestep
```

---

### 8. RT-2 仍然主要是一步一步控制

RT-2 当前 timestep：

```text
Image_t + Instruction
        ↓
       RT-2
        ↓
Action Tokens
        ↓
De-discretization
        ↓
Action_t
```

执行之后：

```text
Action_t
   ↓
新的 Image_(t+1)
   ↓
RT-2
   ↓
Action_(t+1)
```

最终：

```text
a_t → a_(t+1) → a_(t+2) → ...
```

构成完整 trajectory。

因此 RT-2 的核心创新不是 Action Chunk。

---

### 9. Web Data + Robot Data

RT-2 不只使用 Robot Demonstration。

训练时还继续使用原来的 Vision-Language Data：

```text
        RT-2
          ↑
   ┌──────┴──────┐
   ↓             ↓
Web VLM Data   Robot Data
   ↓             ↓
Text Tokens   Action Tokens
```

这样模型可以同时保留：

* 视觉理解
* 语言理解
* 世界知识
* 语义推理

同时学习：

* Robot Action

---

# RT-1 vs RT-2

| 内容                  | RT-1                           | RT-2                  |
| ------------------- | ------------------------------ | --------------------- |
| 核心模型                | Robot Transformer              | Pretrained VLM        |
| Vision              | ✓                              | ✓                     |
| Language            | ✓                              | ✓                     |
| Robot Demonstration | ✓                              | ✓                     |
| Action 离散化          | ✓                              | ✓                     |
| Action Token        | ✓                              | ✓                     |
| Web 世界知识            | 较弱                             | 强                     |
| Action 学习方式         | 监督训练                           | 在 VLM 基础上监督微调         |
| Token Generation    | Robot Action Prediction        | Next-Token Prediction |
| 一次控制                | 一个 timestep                    | 一个 timestep           |
| 核心意义                | Vision + Language Robot Policy | **VLM → VLA**         |

---

# 两天最核心的主线

```text
RT-1
Image History + Language
        ↓
FiLM + EfficientNet
        ↓
TokenLearner
        ↓
Transformer
        ↓
Action Tokens
        ↓
反离散化
        ↓
Action_t
```

```text
RT-2
Pretrained VLM
+
Robot Demonstrations
        ↓
监督微调
        ↓
Image + Language
        ↓
Next-Token Prediction
        ↓
Action Tokens
        ↓
反离散化
        ↓
Action_t
```

## 最终记忆

**RT-1：**

> 用机器人 demonstration 监督训练一个 Vision + Language → Action 的 Robot Policy。

**RT-2：**

> 在已经训练好的 VLM 基础上加入 Robot Demonstration，让 VLM 学会把机器人动作也作为 token 来预测，从而实现 VLM → VLA。

**RT-2 的关键不是 Action Token 本身，因为 RT-1 已经使用离散 Action；真正关键的是将 Action Prediction 接入预训练 VLM 原有的 Next-Token Prediction 框架。**
