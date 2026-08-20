# Week 2 Day 2：π0 学习总结

## 1. π0 的核心思路

π0 仍然保留 VLM 的视觉与语言理解能力，但不再把连续动作主要表示成离散 Action Token，而是加入 **Robot State** 和 **Action Expert**，通过 **Flow Matching** 生成连续的 **Action Chunk**。

整体可以理解为：

```text
Image + Language
      ↓
     VLM
      ↓
视觉与语言语义
      ↓
Robot State
      ↓
Action Expert
      ↓
Flow Matching
      ↓
Continuous Action Chunk
```

核心分工：

- **VLM**：理解“看到了什么、任务是什么、目标物体在哪里”。
- **Robot State**：告诉模型“机器人自己的身体当前是什么状态”。
- **Action Expert**：结合任务语义、Robot State 和当前 noisy action，预测动作应该如何变化。
- **Flow Matching**：把随机噪声逐步变成合理的连续 Action Chunk。

---

## 2. 为什么 π0 要加入 Robot State

图像可以看到机械臂的大致位置和姿态，但图像只能间接估计机器人的自身状态。

Robot State 可以直接提供更加精确的本体信息，例如：

```text
q_t = [q1, q2, q3, ..., qn]
```

其中可以包含当前关节位置等 proprioceptive information。

因此三类输入回答的是三个不同问题：

| 输入 | 作用 |
|---|---|
| Language | 我要做什么？ |
| Image | 外部环境现在是什么样？ |
| Robot State | 我自己的身体现在是什么状态？ |

图像属于外部感知，Robot State 属于本体感觉，两者并不是重复信息。

---

## 3. 为什么 RT / OpenVLA 可以不显式输入 Robot State

早期 RT / OpenVLA 更强调视觉语言泛化和跨机器人训练。

其中一个重要原因是：

不同机器人可能具有不同数量和结构的关节，例如：

```text
Robot A：6 joints
Robot B：7 joints
Robot C：双臂，14 joints
```

如果直接使用原始 joint state 或 joint action：

```text
[q1, q2, q3, ...]
```

不同机器人的维度和物理含义都可能不同，很难直接使用统一接口。

因此跨机器人系统常把动作转换到更统一的末端执行器空间，例如：

```text
[Δx, Δy, Δz,
 Δrx, Δry, Δrz,
 gripper]
```

VLA 主要决定：

> 末端执行器应该往哪里移动。

至于这个 EEF 相对移动最后对应各个 joint 应该怎样变化，是机器人自身运动学和底层控制器负责的问题。

```text
VLA
↓
EEF target / EEF delta
↓
Inverse Kinematics / Robot Controller
↓
Joint Commands
↓
Motor
```

因此可以简单理解为：

> **VLA 管任务空间，机器人底层控制器管关节空间。**

需要注意：不同机器人并不一定都能天然统一成固定 7 维动作。双臂、灵巧手、人形机器人等 embodiment 差异仍然会带来 Action Space 不统一的问题。

---

## 4. Action Expert 是什么

Action Expert 是 π0 中专门处理机器人相关信息和连续动作生成的部分。

它不是简单地把 VLM 输出“翻译”为 Action，也不是：

```text
VLM 先预测 Action
↓
Robot State 再修正
```

更准确的理解是：

```text
Image + Language
      ↓
VLM Features
      │
Robot State
      │
Noisy Action Chunk
      │
      ↓
Action Expert
      ↓
预测当前 Action 应该怎样变化
```

Action Expert 要解决的问题可以理解为：

> 在当前任务、当前环境和当前机器人身体状态下，这条候选动作轨迹应该往哪个方向修改？

因此：

```text
VLM：
我要做什么、目标在哪里

Robot State：
我现在在哪里

Action Expert：
接下来具体应该怎么运动
```

---

## 5. State + VLM 是怎样与 Action 联系起来的

训练数据中本来就存在对应关系：

```text
Image_t
Language
State_t
Action_t
```

或者对于 Action Chunk：

```text
Image_t
Language
State_t
[a_t, a_t+1, ..., a_t+H-1]
```

因此监督数据不断告诉模型：

> 在这种视觉环境、这种语言任务、这种机器人状态下，专家实际执行了怎样的动作轨迹。

模型学习的是条件动作分布：

\[
p(A_t \mid I_t, L, q_t)
\]

其中：

- \(I_t\)：当前图像；
- \(L\)：语言指令；
- \(q_t\)：当前 Robot State；
- \(A_t\)：未来一段 Action Chunk。

Action Expert 通过 attention 使用这些条件信息来生成动作。

---

# 6. π0 的 Flow Matching：训练过程

训练时已经有专家 demonstration，因此知道真实 Action Chunk。

假设：

```text
真实 Action Chunk
A_real = [a_t, a_t+1, ..., a_t+49]
```

训练过程可以理解为：

### Step 1：读取真实训练样本

```text
Image
Language
Robot State
True Action Chunk
```

### Step 2：生成随机 Action Noise

生成和真实 Action Chunk 相同 shape 的随机噪声：

```text
A_noise
```

例如真实 Action Chunk 为：

```text
[50, D]
```

噪声同样是：

```text
[50, D]
```

### Step 3：在 Noise 和真实 Action 之间构造中间状态

随机选择一个 flow time：

\[
\tau \in [0,1]
\]

得到一个介于随机噪声和真实 Action 之间的中间 Action。

可以直观理解为：

```text
Noise
↓
比较乱的 Action
↓
逐渐像真实 Action
↓
True Action
```

### Step 4：Action Expert 预测“应该往哪里变”

输入：

```text
Image
+
Language
+
Robot State
+
Intermediate Action
+
τ
```

Action Expert 输出：

> 当前这个 Action Chunk 应该朝哪个方向变化。

也就是学习一个 action vector field。

### Step 5：计算 Loss

因为训练时知道真实 Action，所以知道正确的变化方向。

比较：

```text
模型预测的变化方向
vs
真实应该变化的方向
```

然后更新模型参数。

训练阶段的核心可以概括为：

```text
真实 Action
+
随机 Noise
↓
构造中间 Action
↓
Image + Language + State + Intermediate Action
↓
Action Expert
↓
预测变化方向
↓
与正确方向计算 Loss
```

---

# 7. π0 的 Flow Matching：推理过程

推理时最大的区别是：

> **没有真实 Action。**

机器人只有：

```text
Image_t
Language
Robot State_t
```

因此需要从随机噪声开始生成动作。

### Step 1：生成随机 Action Noise

```text
A_0 ~ Noise
```

例如：

```text
[50, D]
```

此时完全不能直接执行。

### Step 2：Action Expert 预测变化方向

输入：

```text
Image
Language
Robot State
Noisy Action Chunk
```

Action Expert 根据当前条件判断：

> 这条随机 Action Chunk 应该往哪个方向修改。

### Step 3：不断更新

```text
Random Noise
↓
第一次更新
↓
稍微像合理动作
↓
第二次更新
↓
更加合理
↓
...
↓
Continuous Action Chunk
```

最终从随机 Noise 得到合理的连续 Action Chunk。

---

## 8. Flow Matching 训练和推理的区别

| | 训练 | 推理 |
|---|---|---|
| 是否有真实 Action | 有 | 没有 |
| Image | 有 | 有 |
| Language | 有 | 有 |
| Robot State | 有 | 有 |
| 起点 | 构造 noisy / intermediate Action | 随机 Noise |
| Action Expert | 学习正确变化方向 | 使用学到的方向生成 Action |
| Loss | 有 | 无 |
| 最终目的 | 学会 Action vector field | 生成 Continuous Action Chunk |

可以简单记成：

### 训练

> 已经知道正确 Action，故意构造被扰乱的 Action，让模型学习怎样朝正确 Action 移动。

### 推理

> 不知道正确 Action，因此从随机 Noise 开始，根据训练好的变化方向逐渐生成 Action。

---

# 9. Action Chunk 的 50 步是怎么来的

“50 步”不是人类在示教时主动把任务分成 50 个动作。

人类进行的是连续操作，例如：

> 把手伸过去，抓住杯子。

机器人控制系统会按照固定控制频率持续记录：

```text
t0 → action_0
t1 → action_1
t2 → action_2
...
tT → action_T
```

因此一次 demonstration 最终形成：

```text
a0, a1, a2, ..., aT
```

这就是一条 Action Trajectory。

训练时再从 trajectory 中切出固定长度的窗口。

例如 chunk size = 50：

```text
t = 100

输入：
Image_100
Language
State_100

监督目标：
[a_100, a_101, ..., a_149]
```

下一个时刻：

```text
t = 101

输入：
Image_101
Language
State_101

监督目标：
[a_101, a_102, ..., a_150]
```

因此 Action Chunk 可以看成从完整 trajectory 中截取的一个滑动窗口。

---

## 10. 为什么 Action Chunk 是连续合理的

因为它不是由 50 个互相独立的动作拼起来的。

这些 action 本身来自同一次连续 demonstration：

```text
a_100
a_101
a_102
...
a_149
```

它们天然具有时间连续性。

所以模型学习的是：

\[
p(a_t,a_{t+1},...,a_{t+49}\mid o_t)
\]

而不是只学习：

\[
p(a_t\mid o_t)
\]

因此：

> **Action Chunk 本质上是未来一小段连续 trajectory 的离散数值表示。**

“50”只是人为设定的 horizon / chunk size，并不是任务天然存在 50 个语义步骤。

---

# 11. π0 的整体理解

π0 可以最终概括成：

```text
Image + Language
      ↓
     VLM
      ↓
任务与环境语义
      │
Robot State
      │
      ↓
Action Expert
      ↑
Noisy Action Chunk
      │
      ↓
Flow Matching
      ↓
Continuous Action Chunk
      ↓
Robot Controller
      ↓
Joint Commands
      ↓
Robot Execution
```

核心逻辑：

> **VLM 负责理解任务和环境，Robot State 提供机器人的当前身体状态，Action Expert 结合这些条件，通过 Flow Matching 生成未来一段连续动作轨迹。**
