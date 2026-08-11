# VLA 8 周学习与论文启动计划

> 目标：用 **8 周** 从“对 VLA 体系不熟悉”推进到“能读懂核心模型、能跑通标准 VLA、能修改训练/评测流程，并形成一条可以继续写成论文的研究线”。
>
> 默认投入：**每天约 3 小时，每周 6 天，第 7 天复盘/补缺**。  
> 原则：**不追求把机器人学全部学完；只学习当前实验需要的内容。**
>
> 最终交付不是“看完多少论文”，而是：
>
> 1. 能解释 VLA 的完整输入—模型—动作输出链路；
> 2. 能独立读取 robot dataset；
> 3. 能跑通一个 VLA 在标准 benchmark 上的 inference/evaluation；
> 4. 能完成一次 fine-tuning 或小规模训练；
> 5. 能修改一个研究变量并完成对比实验；
> 6. 第 8 周形成论文问题、实验表格、方法草图和初步结果。

---

# 0. 八周结束时的验收标准

以下 8 项全部完成，才算这轮学习结束。

- [ ] 能解释 observation、state、action、trajectory、episode、demonstration
- [ ] 能解释 Behavior Cloning、Action Chunk、Diffusion Policy、Flow Matching
- [ ] 能解释 OpenVLA 与 π0 的 Action 生成方式有什么不同
- [ ] 能从代码中找到 image、language、state、action、loss 的位置
- [ ] 能独立读取并画出一个 robot episode
- [ ] 能跑通至少一个 VLA 的 LIBERO evaluation
- [ ] 能完成至少一个 VLA 的 fine-tuning / 小规模训练实验
- [ ] 有一份完整的论文实验设计和至少一张有效实验表

---

# 1. 只保留这一条学习主线

不要按照：

```text
Transformer → ViT → CLIP → LLM → VLM → RL → 控制理论 → ROS → VLA
```

学习。

八周只按照：

```text
Robot Data
    ↓
Behavior Cloning
    ↓
Action Chunk / Diffusion
    ↓
VLA
    ↓
OpenVLA / π0 / SmolVLA
    ↓
LIBERO
    ↓
复现
    ↓
修改一个变量
    ↓
论文实验
```

执行。

---

# 2. 推荐实验技术栈

## 主栈

```text
Python
PyTorch
LeRobot
LIBERO
SmolVLA / π0
```

## 第二基线

```text
OpenVLA-OFT
```

## 为什么这样选

### LeRobot
用来统一处理：

- robot dataset
- policy
- training
- evaluation
- LIBERO
- ACT / Diffusion / π0 / π0.5 / SmolVLA 等模型

### LIBERO
前期用它解决两个问题：

1. 不需要真实机器人也能完成 VLA 闭环；
2. 可以快速得到 Success Rate。

但不能把“LIBERO 单一分数提高”直接等同于“通用机器人能力提高”。后期论文至少要增加更严格的拆分、扰动测试或第二类评测。

### SmolVLA
第一阶段推荐模型。

原因：

- 模型相对轻；
- 训练和推理成本低于 7B VLA；
- 适合读代码、改结构、做小规模实验。

### OpenVLA-OFT
作为第二个模型验证方法是否具有跨模型有效性。

---

# 3. 环境初始化

## 第一次只安装 LeRobot

```bash
conda create -n vla python=3.10 -y
conda activate vla
pip install lerobot
lerobot-info
```

如果后续需要修改源码，再切换为源码安装。

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

## 第一次 LIBERO evaluation 的目标

先不要训练。

目标只是确认：

```text
模型可以加载
↓
环境可以启动
↓
图像可以送入模型
↓
模型可以输出 action
↓
机器人可以执行
↓
最终得到 success rate
```

当前 LeRobot 的标准形式类似：

```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

> 注意：实际执行时以你安装版本的官方 README / `--help` 为准，不要机械复制旧教程中的命令。

---

# 4. 第 1 周：Robot Learning 最小基础

## 本周目标

只解决一个问题：

> **VLA 到底在预测什么？**

本周不看 π0 细节。

---

## Day 1：Robot Data 基础

学习：

- observation
- proprioception / state
- action
- episode
- trajectory
- demonstration
- timestep
- control frequency

必须能够解释：

```text
observation_t → policy → action_t
```

### 实际任务

找一个 LeRobot dataset。

打印：

```python
sample.keys()
```

确认至少能找到：

```text
image
state
action
task / instruction
```

### 当日输出

新建：

```text
notes/01_robot_data.md
```

只回答：

1. observation 是什么？
2. state 是什么？
3. action 是什么？
4. 一个 episode 是什么？
5. trajectory 和 episode 有什么区别？

---

## Day 2：机器人 Action

重点理解常见 Action：

```text
joint position
joint velocity
delta joint
end-effector pose
delta end-effector pose
gripper
```

重点：

```text
[x, y, z, rx, ry, rz, gripper]
```

分别代表什么。

### 实际任务

从 dataset 中取一个 episode。

画：

```text
action_dim_1 vs timestep
action_dim_2 vs timestep
...
```

### 当日验收

看到一个 `[T, D]` 的 action tensor 后，可以说清楚：

- T 是什么；
- D 是什么；
- 每一行是什么；
- 为什么相邻 action 通常相关。

---

## Day 3：Behavior Cloning

必学：

```text
expert demonstration
supervised learning
policy
distribution shift
```

核心公式只理解：

```text
π(a_t | o_t)
```

不需要推复杂证明。

### 阅读

**ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**

重点：

- Introduction
- Action Chunking
- Figure 2
- Experiments

### 当日输出

用不超过 300 字解释：

> 为什么机器人模仿学习不能简单理解成普通图像分类？

---

## Day 4：Action Chunk

理解：

```text
输入 observation_t
输出 a_t, a_t+1, ..., a_t+H
```

必须知道：

- chunk size
- horizon
- open-loop execution
- receding horizon
- temporal ensemble

### 实际任务

自己写一个 toy 示例：

```python
obs.shape
actions.shape
```

把单步 action 改成 action chunk。

例如：

```text
单步：
[B, 7]

Action Chunk：
[B, 16, 7]
```

---

## Day 5：Diffusion Policy

阅读：

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**

只理解：

```text
Observation
   ↓
condition
   ↓
noise action trajectory
   ↓
denoising
   ↓
action trajectory
```

重点回答：

1. 为什么机器人 Action 可能是 multimodal？
2. 为什么 MSE regression 可能产生“平均动作”？
3. diffusion 为什么适合生成 action trajectory？

不要求推 diffusion 数学。

---

## Day 6：第一周总复盘

必须独立画出：

```text
Camera
   +
Robot State
   +
Language
   ↓
Policy
   ↓
Action Chunk
   ↓
Robot
```

### 第一周验收

不能看笔记，回答：

- [ ] Behavior Cloning 是什么？
- [ ] Action Chunk 是什么？
- [ ] 为什么一次预测多个 action？
- [ ] state 与 action 有什么区别？
- [ ] Diffusion Policy 和普通 regression 最大区别是什么？

任何一项答不清楚，第 7 天补齐。

---

# 5. 第 2 周：VLM 如何变成 VLA

## 本周目标

回答：

> **VLM 为什么能够控制机器人？**

---

## Day 1：RT-1

阅读：

**RT-1: Robotics Transformer for Real-World Control at Scale**

只关注：

- 输入是什么；
- 输出是什么；
- action 如何表示；
- Transformer 在这里做什么。

输出一张结构图。

---

## Day 2：RT-2

阅读：

**RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**

这是本周最重要的一天。

理解：

```text
VLM:
Image + Language → Text Token

VLA:
Image + Language → Action Token
```

回答：

> 为什么把 Action token 化以后，可以沿用语言模型的 next-token prediction？

---

## Day 3：Open X-Embodiment

重点不看网络结构。

重点看：

```text
不同机器人
不同 action space
不同 camera
不同 control frequency
不同 tasks
```

回答：

> 为什么 cross-embodiment robot learning 比普通多任务学习更难？

---

## Day 4：OpenVLA

阅读：

**OpenVLA: An Open-Source Vision-Language-Action Model**

重点：

```text
Vision Encoder
    ↓
Projector
    ↓
LLM
    ↓
Action Tokens
```

必须理解：

### Action Tokenization

连续动作：

```text
0.032
-0.014
0.008
```

经过离散化后变成 action token。

---

## Day 5：读 OpenVLA 代码

不用全部看。

只找：

```text
1. image 在哪里进入
2. language 在哪里 tokenize
3. action 在哪里 tokenize
4. action ground truth 在哪里
5. loss 在哪里
6. inference 后 action 在哪里 decode
```

### 输出

创建：

```text
notes/02_openvla_code_map.md
```

写成：

```text
Image:
文件：
函数：
shape：

Language:
文件：
函数：

Action:
文件：
函数：

Loss:
文件：
函数：
```

---

## Day 6：第二周复盘

自己解释：

```text
VLM
↓
为什么不能直接控制机器人
↓
加入 action representation
↓
VLA
```

### 第二周验收

- [ ] 能区分 VLM 与 VLA
- [ ] 能解释 action token
- [ ] 能解释 OpenVLA 的输入和输出
- [ ] 能解释 cross-embodiment
- [ ] 能在代码中找到 Action 编解码位置

---

# 6. 第 3 周：现代 VLA——π0、FAST、SmolVLA

## 本周目标

掌握当前最核心的两条 Action 路线：

```text
路线 A：
Action Token → Autoregressive

路线 B：
Continuous Action → Diffusion / Flow Matching
```

---

## Day 1：π0 整体结构

阅读：

**π0: A Vision-Language-Action Flow Model for General Robot Control**

只看：

- Abstract
- Figure 1
- Figure 2
- Architecture
- Training

先建立：

```text
VLM
   ↓
semantic representation
   ↓
Action Expert
   ↓
Flow Matching
   ↓
Continuous Action Chunk
```

---

## Day 2：Flow Matching

只学到“能理解 π0”即可。

必须理解：

```text
noise action
   ↓
velocity field
   ↓
逐步变成真实 action
```

不要花时间推完整数学。

回答：

> π0 为什么不需要像 OpenVLA 一样把每个连续动作维度离散成普通 action token？

---

## Day 3：OpenVLA vs π0

做一张表：

| 项目 | OpenVLA | π0 |
|---|---|---|
| 基础 | VLM | VLM |
| Action | 离散 token | 连续 action |
| 生成 | autoregressive | flow matching |
| 输出 | action tokens | action chunk |
| Action head | token prediction | action expert |

### 当日输出

```text
notes/03_openvla_vs_pi0.md
```

---

## Day 4：FAST

阅读：

**FAST: Efficient Action Tokenization for Vision-Language-Action Models**

重点理解：

> Action tokenization 不等于只能对每个 timestep、每个 dimension 独立分桶。

重点：

```text
action sequence
↓
frequency representation
↓
compressed tokens
↓
autoregressive generation
```

回答：

> FAST 想解决 OpenVLA 式 action tokenization 的什么问题？

---

## Day 5：SmolVLA / OpenVLA-OFT

两篇都只看架构和实验。

### SmolVLA

重点：

- 为什么可以做得更小；
- Action Expert；
- asynchronous inference；
- 哪些参数参与训练。

### OpenVLA-OFT

重点：

- action chunk；
- continuous action head；
- parallel decoding；
- proprioception。

---

## Day 6：第三周总图

独立画：

```text
VLA Action Generation

        ┌─ Discrete token ─ OpenVLA ─ FAST
Image ──┤
Lang  ──┤
State ──┤
        └─ Continuous ─ Diffusion / Flow ─ π0 / SmolVLA
```

### 第三周验收

- [ ] OpenVLA 和 π0 的 Action 生成差异能讲清楚
- [ ] Flow Matching 的直觉能讲清楚
- [ ] FAST 为什么存在能讲清楚
- [ ] Action Expert 是干什么的能讲清楚
- [ ] Action Chunk 与 Action Token 不再混淆

---

# 7. 第 4 周：真正跑通 VLA

## 本周目标

**停止“只看论文”。**

完成：

```text
Dataset
↓
Policy
↓
Inference
↓
Environment
↓
Action
↓
Success Rate
```

---

## Day 1：环境

完成：

```bash
conda activate vla
lerobot-info
```

确认：

```text
CUDA
PyTorch
LeRobot
MuJoCo / LIBERO
```

全部正常。

建立：

```text
experiments/baseline/
```

---

## Day 2：读取 LIBERO 数据

完成：

```text
一个 episode
↓
读取 image
↓
读取 state
↓
读取 action
↓
读取 language instruction
```

保存：

```text
episode_example.png
action_curve.png
```

---

## Day 3：跑 pretrained policy

优先：

```text
π0 / SmolVLA
```

只跑：

```text
LIBERO-Object
10 episodes
```

记录：

```text
success / failure
episode length
inference time
```

---

## Day 4：检查失败案例

随机选：

```text
5 个成功
5 个失败
```

回答：

```text
失败是：
看错目标？
抓取位置错误？
动作不稳定？
执行过慢？
语言理解错误？
长时序积累误差？
```

建立：

```text
results/failure_cases.md
```

---

## Day 5：修改一个简单变量

只改一个：

```text
action chunk size
或
number of camera views
或
proprioception on/off
```

不要追求论文创新。

目的只有一个：

> 证明你已经能够真正修改 VLA 实验，而不是只会运行命令。

---

## Day 6：Baseline 报告

写 2 页以内：

```text
1. Model
2. Dataset
3. Input
4. Output
5. Evaluation
6. Success Rate
7. Failure Cases
8. 一个简单 ablation
```

### 第四周验收

必须有真实结果。

- [ ] 至少一个 pretrained VLA 跑通
- [ ] 至少 10 个 evaluation episodes
- [ ] 有 Success Rate
- [ ] 有失败案例
- [ ] 改过一个模型/输入/动作参数
- [ ] 能定位完整 inference path

如果这里没完成，**禁止进入论文创新阶段。**

---

# 8. 第 5 周：确定论文问题

## 本周目标

从：

> “我在学习 VLA”

切换成：

> “我正在研究 VLA 的一个具体问题。”

---

# 8.1 不建议直接做的题目

不要直接定：

> “VLA demonstration selection”

原因：

2026 年已经有多篇非常接近的工作，例如：

- FrameSkip — `arXiv:2605.13757`
- ATHENA — `arXiv:2606.16208`
- SIEVE — `arXiv:2607.06442`

普通的：

```text
给 trajectory 打分
↓
选 top 50%
↓
训练 VLA
↓
优于 random
```

已经很容易撞题。

---

# 8.2 优先考虑的三个方向

## 方向 A：Action-critical Multimodal Data Valuation

### 核心问题

不是：

> 哪条 trajectory 更好？

而是：

> **哪些 observation / segment 中的视觉或语言信息真正改变了 Action prediction？**

可以研究：

```text
Full:
Image + Language + State → Action

Remove Image:
Language + State → Action

Remove Language:
Image + State → Action

Remove State:
Image + Language → Action
```

观察 Action Loss / Action Error 的变化。

形成：

```text
Visual Contribution
Language Contribution
State Contribution
Action Criticality
```

再用于：

```text
frame / segment weighting
或
data curriculum
或
sample selection
```

### 优点

- 与普通 trajectory selection 区别更明显；
- 能形成机制分析；
- 可以做跨任务比较；
- 可以验证不同 VLA 是否依赖不同模态。

### 缺点

- 必须认真检查 2025–2026 最新工作；
- Flow-based policy 的 scoring 需要设计得严谨。

### 推荐度

**★★★★★**

---

## 方向 B：Adaptive Action Chunk

### 核心问题

当前很多方法固定：

```text
chunk size = H
```

但不同动作阶段：

```text
free-space movement
grasp
contact
placement
```

对时间精度需求不同。

研究：

```text
简单运动 → long chunk
精细操作 → short chunk
```

### 优点

- 问题直接；
- 容易解释；
- Action 侧研究价值高；
- 与 π0 / SmolVLA / OpenVLA-OFT 都相关。

### 缺点

- 代码改动比数据分析更大；
- inference/evaluation 要严谨。

### 推荐度

**★★★★☆**

---

## 方向 C：VLA Evaluation Robustness

### 核心问题

不只问：

```text
LIBERO Success Rate 是否提高
```

而是：

```text
初始位置扰动
目标位置扰动
instruction paraphrase
camera perturbation
background perturbation
```

之后还是否提高。

### 优点

- 算力成本较低；
- 容易快速形成大量结果；
- 可以和模型机制分析结合。

### 缺点

- 单纯“测 benchmark”创新不足；
- 必须提出新的诊断指标或系统性发现。

### 推荐度

**★★★☆☆**

---

# 8.3 第五周决策规则

按以下顺序判断：

```text
是否已经有几乎相同论文？
        ↓
       是 → 换
        ↓
       否
        ↓
能否在 2 周内做 pilot？
        ↓
       否 → 缩小
        ↓
       是
        ↓
是否至少有 2 个 baseline？
        ↓
       否 → 重构
        ↓
       是
        ↓
正式确定课题
```

---

## Day 1–2：文献矩阵

建立：

```text
paper_matrix.xlsx / paper_matrix.md
```

至少 15 篇近年论文。

字段：

```text
Paper
Year
Model
Dataset
Benchmark
Research Question
Method
Baseline
Metric
Main Conclusion
Limitation
与我的区别
```

---

## Day 3：写 Research Question

只能一句话。

格式：

> Existing VLA methods ______, but ______ remains unclear.

然后写：

```text
RQ1
RQ2
RQ3
```

最多 3 个。

---

## Day 4：实验矩阵

例如：

| Group | Data/Method | Model | Benchmark | Metric |
|---|---|---|---|---|
| B1 | Full | SmolVLA | LIBERO | SR |
| B2 | Random 50% | SmolVLA | LIBERO | SR |
| M1 | Proposed | SmolVLA | LIBERO | SR |
| M2 | Proposed | OpenVLA-OFT | LIBERO | SR |

---

## Day 5：Pilot

只做：

```text
1 个 task suite
1 个 baseline
1 个 proposed method
```

不要全量跑。

目标：

> 看有没有明显信号。

---

## Day 6：Go / No-Go

### Go

满足：

- 方法能运行；
- 结果不是明显负面；
- 有机制可解释；
- 与已有论文有明确区别。

### No-Go

如果：

- 与已有方法高度重复；
- 实验成本超出可接受范围；
- 两组结果完全无差异；
- 指标无法稳定复现；

立即换方向，不在错误方向继续耗两周。

---

# 9. 第 6 周：第一轮完整实验

## 本周目标

得到论文第一张正式主表。

---

## 实验原则

一次只改一个主要变量。

所有实验固定：

```text
model
dataset split
training steps
batch size
seed
evaluation episodes
```

除非该变量本身就是研究对象。

---

## Day 1：Baseline

至少：

```text
Base model
Random baseline
一个已有方法 baseline
```

---

## Day 2–3：主方法

完成：

```text
Proposed Method
```

至少在：

```text
LIBERO-Spatial
LIBERO-Object
```

上运行。

算力足够再加入：

```text
LIBERO-Goal
LIBERO-10
```

---

## Day 4：重复实验

关键结果至少多 seed / 多次 evaluation。

不要只记录一次成功率。

记录：

```text
mean
std
number of episodes
```

---

## Day 5：第一张主表

目标形式：

| Method | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|
| Baseline | | | | | |
| Random | | | | | |
| Proposed | | | | | |

---

## Day 6：结果判断

只回答：

1. 是否有效？
2. 在什么任务有效？
3. 在什么任务无效？
4. 为什么？
5. 有没有反例？

如果只能说：

> “平均提高了 2%”

还不够。

---

# 10. 第 7 周：Ablation + Mechanism

## 本周目标

回答：

> **为什么有效？**

这决定论文是否只是一个小 trick。

---

## 必须至少做 3 类分析

### 1. Ablation

例如：

```text
without visual score
without language score
without action score
full
```

---

### 2. Data Ratio / Parameter Sensitivity

例如：

```text
25%
50%
75%
100%
```

或者：

```text
chunk = 4
chunk = 8
chunk = 16
```

---

### 3. Failure Analysis

至少分类：

```text
Perception failure
Language grounding failure
Motion failure
Grasp failure
Long-horizon failure
```

---

## 推荐增加：扰动评测

例如：

```text
instruction paraphrase
object position shift
camera perturbation
```

原因：

仅提高标准 LIBERO success rate 的说服力有限。

---

## Day 1–2

Ablation。

## Day 3

Sensitivity。

## Day 4

Failure cases。

## Day 5

Visualization。

至少准备：

```text
Figure 1：Method
Figure 2：Main result
Figure 3：Ablation / analysis
Figure 4：Failure / qualitative
```

## Day 6

根据结果重新写：

```text
Research Question
Hypothesis
Main Finding
```

---

# 11. 第 8 周：形成论文雏形

## 本周目标

不是“马上投稿”。

目标是：

> **已经拥有一篇可以继续扩充到投稿状态的 paper skeleton。**

---

## Day 1：确定故事

只写四句话：

```text
Problem:
现有方法的问题是什么？

Observation:
你发现了什么？

Method:
你怎么解决？

Result:
结果说明什么？
```

如果四句话说不清楚，说明论文主线还不清楚。

---

## Day 2：写 Introduction 骨架

只写四段：

```text
P1：VLA 重要性
P2：当前问题
P3：现有方法不足
P4：本文方法 + contributions
```

不要一开始追求语言。

---

## Day 3：Methods

必须画出：

```text
Input
↓
VLA
↓
你的模块 / scoring / action strategy
↓
Action
```

并写：

```text
Problem Formulation
Method Overview
核心公式
Training / Inference
```

---

## Day 4：Experiments

写完整：

```text
Datasets
Benchmarks
Baselines
Implementation
Metrics
Main Results
```

即使部分数字后续还要补，也先确定表格。

---

## Day 5：Results + Analysis

每张表必须对应一个问题。

禁止：

```text
Table 1：堆数字
Table 2：再堆数字
Table 3：继续堆数字
```

应该是：

```text
Table 1：方法是否有效？
Table 2：为什么有效？
Table 3：数据少时是否仍有效？
Table 4：换模型以后是否仍有效？
```

---

## Day 6：八周最终评审

必须具有以下文件：

```text
VLA_PROJECT/
│
├── notes/
│   ├── 01_robot_data.md
│   ├── 02_openvla_code_map.md
│   └── 03_openvla_vs_pi0.md
│
├── literature/
│   └── paper_matrix.md
│
├── experiments/
│   ├── baseline/
│   ├── proposed/
│   └── ablation/
│
├── results/
│   ├── main_results.csv
│   ├── ablation.csv
│   ├── failure_cases.md
│   └── figures/
│
└── paper/
    ├── outline.md
    ├── intro.md
    ├── method.md
    └── experiments.md
```

---

# 12. 八周论文阅读清单

## A 类：必须精读

按顺序：

1. **ACT**  
   Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware  
   `arXiv:2304.13705`

2. **Diffusion Policy**  
   Visuomotor Policy Learning via Action Diffusion  
   `arXiv:2303.04137`

3. **RT-2**  
   Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

4. **OpenVLA**  
   OpenVLA: An Open-Source Vision-Language-Action Model  
   `arXiv:2406.09246`

5. **π0**  
   A Vision-Language-Action Flow Model for General Robot Control  
   `arXiv:2410.24164`

6. **FAST**  
   Efficient Action Tokenization for Vision-Language-Action Models  
   `arXiv:2501.09747`

7. **OpenVLA-OFT**  
   Fine-Tuning Vision-Language-Action Models  
   `arXiv:2502.19645`

8. **SmolVLA**  
   A Vision-Language-Action Model for Affordable and Efficient Robotics  
   `arXiv:2506.01844`

---

## B 类：理解即可

- RT-1
- Open X-Embodiment
- Octo
- π0.5
- LeRobot paper

---

## C 类：第 5 周选题时精读

如果做数据相关：

- FrameSkip — `arXiv:2605.13757`
- ATHENA — `arXiv:2606.16208`
- SIEVE — `arXiv:2607.06442`

目的是确认：

> 你的问题、score、粒度、实验设置与已有工作到底有什么不同。

---

# 13. 每篇论文只回答这 8 个问题

不要逐句翻译论文。

统一使用：

```markdown
# Paper

## 1. Problem
它解决什么问题？

## 2. Previous Limitation
以前为什么做不好？

## 3. Input
模型输入什么？

## 4. Output
模型输出什么？

## 5. Action Representation
Action 怎么表示？

## 6. Method
核心方法是什么？

## 7. Experiment
用什么数据、benchmark、metric？

## 8. What I Can Use
对我的实验有什么直接价值？
```

单篇论文笔记控制在 **1–2 页**。

---

# 14. 代码阅读统一检查表

每次读新 VLA，只找下面这些。

```text
[ ] Dataset loader
[ ] Image preprocessing
[ ] Language tokenizer
[ ] Robot state preprocessing
[ ] Action normalization
[ ] Action representation
[ ] VLM backbone
[ ] Action head / action expert
[ ] Loss
[ ] Action decoding
[ ] Action chunk
[ ] Evaluation loop
```

如果这 12 个位置都找到，才叫“读过这个模型代码”。

---

# 15. 每周时间分配

默认每天 3 小时。

## 第 1–3 周

```text
论文：1 h
代码：1 h
总结/画图：1 h
```

## 第 4 周以后

```text
论文：0.5 h
代码/实验：2 h
实验记录：0.5 h
```

第 5 周以后禁止每天大量刷论文。

文献只为解决：

```text
这个问题有人做了吗？
baseline 是什么？
实验应该怎么设计？
```

---

# 16. 实验记录模板

每次实验都记录。

```markdown
# Experiment ID

## Hypothesis
为什么做？

## Change
只改了什么？

## Fixed
哪些条件完全不变？

## Config
Model:
Dataset:
Seed:
Steps:
Batch Size:
Chunk Size:

## Result
Success Rate:
Mean:
Std:

## Observation
发生了什么？

## Conclusion
支持还是否定 hypothesis？

## Next
下一步唯一需要做什么？
```

没有记录的实验视为没做。

---

# 17. 论文方向选择：最终推荐

如果目标是 **8 周内最快进入可发表研究状态**，优先级建议：

| 方向 | 上手速度 | 算力 | 2026 撞题风险 | 机制深度 | 推荐 |
|---|---:|---:|---:|---:|---:|
| 普通 trajectory selection | 快 | 低 | 很高 | 中 | ★★ |
| Action-critical multimodal valuation | 中 | 中 | 中 | 高 | ★★★★★ |
| Adaptive action chunk | 中 | 中 | 中 | 高 | ★★★★ |
| 新 VLA backbone | 慢 | 高 | 中 | 高 | ★★ |
| 纯 benchmark 测试 | 快 | 低 | 中 | 低 | ★★★ |

## 推荐主线

优先探索：

> **Action-critical multimodal data valuation for VLA fine-tuning**

核心不是简单挑“高质量数据”，而是研究：

```text
视觉
语言
机器人状态
```

在一个 robot trajectory 的不同阶段，对 **Action prediction** 到底贡献多少。

这条线能够同时连接：

```text
Data
+
Multimodal Understanding
+
Action
```

并且容易自然形成：

```text
Main Result
Data Efficiency
Modality Analysis
Task Analysis
Cross-model Validation
Failure Analysis
```

但 **第 5 周做完最新文献检查之前，不锁死题目名称。**

---

# 18. 8 周期间禁止做的事情

- 不从头学完整机器人运动学教材
- 不从头学完整控制理论
- 不为了 VLA 先系统学习 RL
- 不一开始训练 foundation model
- 不同时跑 5 个 VLA
- 不同时研究数据、结构、RL、world model
- 不因为一篇新论文出现就立刻换方向
- 不只看 loss，不做 robot rollout
- 不只报告单次 success rate
- 不把 benchmark 提升自动解释成 generalization 提升
- 不在没有 baseline 的情况下写“创新方法”

---

# 19. 最短执行版

如果某一周很忙，只保留下列任务。

## Week 1

```text
ACT
Diffusion Policy
读 Robot Dataset
```

## Week 2

```text
RT-2
OpenVLA
读 OpenVLA code
```

## Week 3

```text
π0
FAST
SmolVLA
```

## Week 4

```text
LeRobot
LIBERO
跑通 pretrained VLA
```

## Week 5

```text
最新文献矩阵
确定 Research Question
完成 pilot
```

## Week 6

```text
Baseline
Main Experiment
第一张主表
```

## Week 7

```text
Ablation
Mechanism
Failure Analysis
```

## Week 8

```text
Figure
Paper Outline
Introduction
Methods
Experiments
```

---

# 20. 最终判断

八周以后，你不需要达到：

> “我已经掌握全部 VLA。”

真正需要达到：

> **我知道 VLA 的 Action 是怎么来的，我能跑、能改、能测，并且已经围绕一个明确问题拿到了第一轮研究结果。**

这就是从“学习 VLA”进入“做 VLA 研究”的分界线。
