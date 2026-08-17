# VLA → 类脑具身智能 8周学习计划：飞书表格版（NeuroVLA 组内版）

## 总目标

| 项目 | 内容 |
|---|---|
| 8周定位 | 主流 VLA 建立参照系 → 类脑基础 → NeuroVLA 深入理解与复现 → 形成类脑具身研究问题 |
| 默认投入 | 每天约 3 小时，每周 6 天，第 7 天复盘/补缺 |
| 主流 VLA 学到什么程度 | 能看懂、能比较、能跑，不追求把 π 系列全部精读 |
| 类脑部分学到什么程度 | 能解释机制、定位代码、改模块、做对照实验 |
| 最终交付 | NeuroVLA 结构图 + 代码地图 + baseline/ablation + 类脑研究问题 + pilot + 论文骨架 |

## 8周总览

| 周次 | 阶段目标 | 核心问题 | 主要任务 | 必须产出 |
|---|---|---|---|---|
| Week 1 | Robot Learning 基础 | VLA 到底在预测什么？ | Robot Data、Action、BC、Action Chunk、Diffusion | `notes/01_robot_data.md` |
| Week 2 | VLM → VLA | Vision、Language、Action 怎么连起来？ | RT-1、RT-2、OpenVLA、Open X-Embodiment、融合层次 | `notes/02_rt1_rt2.md` + RT-1结构图 |
| Week 3 | 主流 VLA 参照系 | 普通 VLA 的主要瓶颈是什么？ | π0、Flow Matching、π0.5、RTC、Memory/Online Learning | `notes/03_vla_bottlenecks.md` |
| Week 4 | 类脑计算基础 | SNN 为什么适合机器人？ | LIF、Spike、Surrogate Gradient、STDP、R-STDP | `notes/04_neuromorphic_basics.md` |
| Week 5 | NeuroVLA 精读 | Cortex–Cerebellum–Spinal 为什么这样拆？ | 论文、代码、模块、Ablation | `notes/05_neurovla_code_map.md` |
| Week 6 | NeuroVLA 复现 | 每一层是否真的有作用？ | Baseline、No-Cerebellum、Single/Multi-step SNN、Perturbation | `results/neurovla_baseline.md` |
| Week 7 | 研究问题 | 下一步类脑创新点在哪里？ | 文献矩阵、RQ、实验矩阵、pilot、Go/No-Go | `literature/brain_vla_matrix.md` + pilot |
| Week 8 | 论文雏形 | 能否讲清楚“生物机制→方法→机器人收益”？ | 主图、主表、ablation、failure、outline | `paper/outline.md` + method figure |

## Week 1：Robot Learning 最小基础

| 天 | 主题 | 学习重点 | 当日产出 / 验收 |
|---|---|---|---|
| Day 1 | Robot Data | observation、state、action、episode、trajectory、demonstration | 能解释完整数据链 |
| Day 2 | Robot Action | joint/EE/delta action、`[T,D]` | 能解释 state vs action |
| Day 3 | Behavior Cloning | `π(a_t|o_t)`、distribution shift | 能解释 BC 与普通分类差别 |
| Day 4 | Action Chunk | chunk、horizon、receding horizon、temporal ensemble | 能解释为什么一次预测多个 action |
| Day 5 | Diffusion Policy | multimodal action、noise、denoising、trajectory | 能解释 diffusion 为什么适合 action |
| Day 6 | 复盘 | Camera + State + Language → Policy → Action | 不看笔记完成总图 |

## Week 2：VLM 怎么变成 VLA

| 天 | 主题 | 学习重点 | 当日产出 / 验收 |
|---|---|---|---|
| Day 1 | RT-1 | 6帧 image、USE、FiLM、EfficientNet、TokenLearner、Transformer | RT-1 结构图 |
| Day 2 | RT-2 | VLM→VLA、action token、next-token prediction | `notes/02_rt1_rt2.md` |
| Day 3 | OpenVLA | Vision Encoder、Projector、LLM、Action Tokens | 能解释 action tokenization |
| Day 4 | Open X-Embodiment | embodiment、action space、camera、frequency | 能解释 cross-embodiment |
| Day 5 | 融合方式 | FiLM、Cross-attention、CLIP-style alignment | 能解释融合层次影响 |
| Day 6 | 复盘 | Image + Language + State → Representation → Action | VLA 总图 |

## Week 3：现代 VLA 最小参照系

| 天 | 主题 | 学习重点 | 当日产出 / 验收 |
|---|---|---|---|
| Day 1 | π0 | VLM、Action Expert、Flow Matching、Continuous Action Chunk | π0 结构图 |
| Day 2 | Flow Matching | noise action、velocity field、continuous action | 能解释 π0 vs OpenVLA |
| Day 3 | π0.5 | open-world generalization、多来源数据 | 能解释泛化问题 |
| Day 4 | Real-Time Action Chunking | VLA latency、连续执行、stale action | 能解释实时控制矛盾 |
| Day 5 | Memory / Online Learning | 长短期记忆、deployment 后学习 | 能解释为什么需要持续适应 |
| Day 6 | VLA Bottleneck Map | latency、feedback、jitter、memory、energy、online learning | `notes/03_vla_bottlenecks.md` |

## Week 4：类脑计算基础

| 天 | 主题 | 学习重点 | 当日产出 / 验收 |
|---|---|---|---|
| Day 1 | LIF | membrane、threshold、leak、reset | 能解释 SNN 的时间状态 |
| Day 2 | Spike Representation | rate / temporal coding、event-driven、sparsity | 画 sensor→spike toy 图 |
| Day 3 | SNN Training | surrogate gradient、BPTT | 能解释 spike 不可导怎么训练 |
| Day 4 | STDP | pre/post timing、local plasticity | 能解释局部学习 |
| Day 5 | R-STDP | reward、eligibility、reward modulation | 能解释在线适应价值 |
| Day 6 | 类脑控制总图 | slow cognition / adaptive feedback / fast reflex | `notes/04_neuromorphic_basics.md` |

## Week 5：NeuroVLA 精读 + 代码地图

| 天 | 主题 | 学习重点 | 当日产出 / 验收 |
|---|---|---|---|
| Day 1 | Overall | Cortex–Cerebellum–Spinal、多时间尺度 | 自画 NeuroVLA 图 |
| Day 2 | Cortex | VLM、Q-Former、intention extraction | 能解释 high-level intention |
| Day 3 | Cerebellum | proprioception、force、gain modulation、smoothing | 能解释 adaptive feedback |
| Day 4 | Spinal SNN | event-driven、temporal integration、reflex、hardware | 能解释为什么不是普通 action head |
| Day 5 | 代码地图 | Dataset、Cortex、Cerebellum、Spinal、Loss、Eval | `notes/05_neurovla_code_map.md` |
| Day 6 | Ablation | No-Cerebellum、Single/Multi-step SNN、latency、jerk、energy | 能解释每层贡献 |

## Week 6：真正跑 NeuroVLA

| 天 | 主题 | 实验 | 验收 |
|---|---|---|---|
| Day 1 | Baseline | LIBERO / 组内 benchmark / 真机标准任务 | success、latency、episode |
| Day 2 | Cerebellum Ablation | 去掉 Cerebellum | 比较 success / jerk / stability |
| Day 3 | SNN Temporal Ablation | Single-step vs Multi-step | 验证 temporal state |
| Day 4 | Frequency | 改 Cortex/Cerebellum/Spinal update frequency | 多时间尺度结果 |
| Day 5 | Perturbation | visual noise / force / collision / delay 选一个 | robustness 结果 |
| Day 6 | Baseline 报告 | 汇总方法、指标、失败案例 | `results/neurovla_baseline.md` |

## Week 7：研究方向选择

| 方向 | 核心问题 | 推荐度 |
|---|---|---|
| Adaptive Fast–Slow Routing | 根据任务阶段/风险/不确定性动态决定快慢通路 | ★★★★★ |
| Online Cerebellar/Spinal Plasticity | 低层回路能否通过局部反馈在线适应 | ★★★★★ |
| Multi-timescale Motor Memory | 不同时间尺度的运动记忆应该存在哪里 | ★★★★☆ |
| Event/Force/Tactile Reflex | 高频事件是否应绕过大 VLA 直接进入 fast loop | ★★★★☆ |
| Action-critical Data Valuation | 哪些模态片段真正改变 action | ★★★ |

### Week 7 每日安排

| 天 | 任务 | 验收 |
|---|---|---|
| Day 1-2 | 建立 brain-inspired VLA 文献矩阵 | 4类文献：real-time、neuromorphic、online learning、motor memory |
| Day 3 | 写最多3个 RQ | 每个 RQ 能对应明确实验 |
| Day 4 | 设计实验矩阵 | NeuroVLA + conventional VLA + ablation + proposed |
| Day 5 | Pilot | 1 task + 1 baseline + 1 change |
| Day 6 | Go / No-Go | 有机制、有量化指标、不是简单调参 |

## Week 8：论文雏形

| 天 | 任务 | 必须回答 |
|---|---|---|
| Day 1 | 四句话 | Problem / Biological Inspiration / Method / Result |
| Day 2 | Method Figure | 新机制加在 Cortex、Cerebellum 还是 Spinal？ |
| Day 3 | Main Result | 是否改善 success / latency / jerk / robustness / energy |
| Day 4 | Mechanism / Ablation | 为什么有效？ |
| Day 5 | Failure Analysis | 哪一层失败？ |
| Day 6 | Final Review | 是否已经形成可继续扩展的 paper skeleton |

## π 系列阅读优先级

| 工作 | 建议 |
|---|---|
| π0 | 精读 |
| π0.5 | 重点读 |
| Real-Time Action Chunking | 精读 |
| Memory | 精读思想 |
| π0.6 / online RL | 选读 |
| FAST | 快速读 |
| π0.7 | 后读 |

## 最终定位

不是：

> 把所有 VLA 论文看完。

而是：

> **能用主流 VLA 作为参照，准确解释 NeuroVLA 为什么需要多时间尺度的 Cortex–Cerebellum–Spinal 架构，并在这个架构上提出一个新的、可验证的类脑具身研究问题。**
