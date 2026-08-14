# Day One：Robot Data 基础

## 今日目标

理解 VLA 最基本的数据链路：

```text
observation_t → policy → action_t
```

也就是：模型读取当前时刻的观测，经过策略模型计算，输出机器人需要执行的动作。

## 核心概念

### 1. Observation

`observation` 是模型在当前时刻能够获得的全部信息，例如：

- 相机图像；
- 机器人自身状态；
- 任务或语言指令。

Observation 不只是图像，state 通常也是 observation 的一部分。

### 2. State / Proprioception

`state` 表示机器人自身当前的状态，例如关节角、关节速度、末端执行器位置、夹爪状态等。

在 PushT 数据集中，`observation.state` 是一个二维向量：

```text
shape = [2]
```

它描述当前二维环境中的机器人或推动器状态。

### 3. Action

`action` 是策略模型输出给机器人或环境的控制命令，而不是动作执行后的结果。

在 PushT 数据集中，action 是一个二维向量：

```text
shape = [2]
```

它表示二维环境中的控制动作。它的具体物理含义需要结合数据集和环境定义判断。

### 4. Timestep

一个 `timestep` 表示一次完整的控制时刻：

```text
读取 observation → 预测 action → 执行 action
```

数据集中的一个 sample 通常表示一个 timestep，而不是完整的 episode。

### 5. Control Frequency

`control frequency` 表示系统每秒执行多少次控制。

PushT 数据集的频率为：

```text
10 Hz
```

因此相邻两个 timestep 的理论时间间隔为：

```text
1 / 10 = 0.1 秒
```

### 6. Trajectory

`trajectory` 是按照时间顺序排列的观测和动作序列：

```text
(o_0, a_0), (o_1, a_1), ..., (o_T, a_T)
```

它强调数据随时间变化的过程。

### 7. Episode

`episode` 是一次有明确开始和结束的完整任务执行。它可能因为任务成功、失败或超时而结束。

一个 episode 通常包含一条 trajectory，以及任务描述、成功标记等额外信息。

### 8. Demonstration

`demonstration` 是用于教模型完成任务的示范轨迹，通常由人类操作者或专家策略产生。

Demonstration 是一种具有训练用途的 trajectory。

## 今日使用的数据集

数据集：

```text
lerobot/pusht
```

自然语言任务：

```text
Push the T-shaped block onto the T-shaped target.
```

中文含义：把 T 形物块推到 T 形目标区域上。

数据集基本信息：

```text
Episode 数量：206
Sample 数量：25650
控制频率：10 Hz
```

本地缓存位置：

```text
C:\Users\77170\.cache\huggingface\lerobot\lerobot\pusht
```

## 样本字段

运行 `sample.keys()` 得到的主要字段如下：

| 字段 | Shape | 含义 |
| --- | --- | --- |
| `observation.image` | `[3, 96, 96]` | 当前时刻的 RGB 图像 |
| `observation.state` | `[2]` | 当前时刻的二维状态 |
| `action` | `[2]` | 当前时刻对应的二维动作 |
| `episode_index` | 标量 | 所属 episode 的编号 |
| `frame_index` | 标量 | 当前帧在 episode 中的编号 |
| `timestamp` | 标量 | 当前 timestep 的时间戳 |
| `next.reward` | 标量 | 执行动作之后得到的奖励 |
| `next.done` | 标量 | 下一步是否结束 episode |
| `next.success` | 标量 | 任务是否成功 |
| `index` | 标量 | 样本在整个数据集中的全局编号 |
| `task_index` | 标量 | 任务描述的编号 |
| `task` | 字符串 | 自然语言任务指令 |

数据集元数据中的图像 shape 是 `[96, 96, 3]`，表示 `HWC`；读取为 PyTorch tensor 后是 `[3, 96, 96]`，表示 `CHW`。

## 数据链路

在当前数据集中，可以把输入和输出关系理解为：

```text
observation.image
        +
observation.state
        +
task
        ↓
      policy
        ↓
      action
```

模型根据图像、当前状态和任务要求，预测下一步应该执行的动作。

## 五个问题

### 1. Observation 是什么？

Observation 是模型在当前时刻能够获得的全部信息，可以包含相机图像、机器人状态和任务指令等。

### 2. State 是什么？

State 是描述机器人自身当前状态的数值信息，通常是 observation 的一部分。

### 3. Action 是什么？

Action 是策略模型输出给机器人控制器或环境的控制命令，其具体含义取决于 action space。

### 4. 一个 Episode 是什么？

Episode 是从任务开始到成功、失败或超时结束的一次完整交互过程。

### 5. Trajectory 和 Episode 有什么区别？

Trajectory 强调按照时间排列的状态和动作序列；episode 强调一次具有明确开始、结束和任务结果的完整运行。

## 运行时警告

本次运行出现了两个不影响使用的警告：

- 没有安装 `torchcodec`，LeRobot 自动使用 `pyav` 解码视频；
- 没有安装 `hf_xet`，Hugging Face 自动使用普通 HTTP 下载文件。

它们不会影响本次数据读取任务，因此暂时不需要处理。

## 今日完成情况

- [x] 安装并成功导入 LeRobot 0.4.4
- [x] 下载并加载 `lerobot/pusht`
- [x] 打印数据集 features
- [x] 打印 `sample.keys()`
- [x] 找到 image、state、action 和 task
- [x] 确认 episode、frame、timestamp 和成功标记字段
- [x] 确认数据集的本地缓存位置
