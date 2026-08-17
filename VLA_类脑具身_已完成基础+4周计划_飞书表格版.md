# VLA → 类脑具身智能：已完成基础 + 4周计划（飞书表格版）


## 已完成基础（原 Week 1，不占后续 4 周）

| 模块 | 已学习内容 | 状态 |
|---|---|---|
| Robot Data | observation、state、action、episode、trajectory、demonstration、timestep、control frequency | ✅ |
| Robot Action | joint / EE / delta action、gripper、`[T,D]`、state vs action | ✅ |
| Behavior Cloning | expert demonstration、`π(a_t \| o_t)`、distribution shift | ✅ |
| Action Chunk | chunk size、horizon、open-loop、receding horizon、temporal ensemble | ✅ |
| Diffusion Policy | multimodal action、MSE 平均、noise / denoising、action trajectory | ✅ |
| 综合 | Camera + State + Language → Policy → Action Chunk → Robot | ✅ |

> 后续 4 周直接从 RT-1 开始，不重复上述基础。


## 4周总览

| 周次 | 目标 | 核心问题 | 主要任务 | 必须产出 |
|---|---|---|---|---|
| Week 1 | 普通 VLA 学到够用 | VLA 怎么从视觉语言生成 action？还缺什么？ | RT-1、RT-2、OpenVLA、π0、VLA bottleneck map | `vla_gap_map_v0.md` |
| Week 2 | 类脑基础 + NeuroVLA | 为什么需要 Cortex–Cerebellum–Spinal？ | LIF、SNN、STDP/R-STDP、NeuroVLA 架构与 ablation | `neurovla_architecture.md` + `neurovla_mechanism.md` |
| Week 3 | 代码 + 复现 + Gap | NeuroVLA 哪一块还能改？ | 代码地图、baseline、1个 ablation、gap map、定向文献 | `neurovla_code_map.md` + `gap_map_v1.md` |
| Week 4 | Pilot + 决策 | 这个 gap 值不值得继续做？ | RQ、实验矩阵、pilot、Go/No-Go | `research_question.md` + pilot结果 |

## Week 1

| 天 | 主题 | 重点 | 验收 |
|---|---|---|---|
| Day 1 | RT-1 | FiLM、EfficientNet、TokenLearner、Transformer、Action | 能完整讲输入到 Action |
| Day 2 | RT-2 | VLM→VLA、Action token | 能解释 next-token prediction |
| Day 3 | OpenVLA | Vision Encoder、Projector、LLM、Action Token | 能解释离散 action |
| Day 4 | π0 | Action Expert、Flow Matching、Continuous Chunk | 能解释 π0 vs OpenVLA |
| Day 5 | VLA Bottleneck | latency、feedback、memory、online learning、energy | `vla_gap_map_v0.md` |
| Day 6 | 复盘 | 普通 VLA 能力与不足 | 能回答为什么 VLM 更大不等于运动控制更好 |

## Week 2

| 天 | 主题 | 重点 | 验收 |
|---|---|---|---|
| Day 1 | LIF / SNN | temporal state、spike、sparsity | 能解释 SNN 时间特性 |
| Day 2 | Surrogate Gradient / STDP | spike training、local plasticity | 能解释局部学习 |
| Day 3 | R-STDP | reward、eligibility、online adaptation | 能解释在线适应 |
| Day 4 | NeuroVLA Overall | Cortex–Cerebellum–Spinal | 自画架构图 |
| Day 5 | NeuroVLA Modules | 输入输出、时间尺度、feedback | `neurovla_architecture.md` |
| Day 6 | Ablation | No-Cerebellum、Single/Multi-step SNN、latency/jerk | `neurovla_mechanism.md` |

## Week 3

| 天 | 主题 | 任务 | 验收 |
|---|---|---|---|
| Day 1 | Code Map | 定位 Dataset、Cortex、Cerebellum、Spinal、Loss、Eval | `neurovla_code_map.md` |
| Day 2 | Baseline | 跑组内 benchmark / checkpoint | success + latency + failure |
| Day 3 | Ablation | 复现 1 个模块消融 | 确认能真正修改 NeuroVLA |
| Day 4 | Gap Map | 列出已解决与未解决问题 | `gap_map_v1.md` |
| Day 5 | 定向文献 | 只查 2 个候选 gap | 不再泛读 VLA |
| Day 6 | 候选方向 | Fast–Slow Routing / Online Plasticity / Motor Memory / Reflex | 选 1–2 个 |

## Week 4

| 天 | 主题 | 任务 | 验收 |
|---|---|---|---|
| Day 1 | RQ | 一句话 research question + hypothesis | RQ 明确 |
| Day 2 | 实验矩阵 | Conventional VLA / NeuroVLA / Ablation / Proposed | 最小实验设计 |
| Day 3-4 | Pilot | 1 task + 1 baseline + 1 proposed | 初步结果 |
| Day 5 | 分析 | 判断机制、收益、反例 | 不是只看 success rate |
| Day 6 | Go / No-Go | 决定是否继续 | `research_question.md` |

## 研究方向优先级

| 方向 | 推荐度 |
|---|---|
| Adaptive Fast–Slow Routing | ★★★★★ |
| Online Cerebellar / Spinal Plasticity | ★★★★★ |
| Multi-timescale Motor Memory | ★★★★☆ |
| Event / Force / Tactile Fast Reflex | ★★★★☆ |

## 四周结束标准

> **能够准确说出普通 VLA 的瓶颈、NeuroVLA 已经解决的部分、一个仍未解决的类脑具身 gap，并且已经完成一个最小 pilot。**
