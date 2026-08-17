# VLA → 类脑具身智能 8 周学习与论文启动计划（NeuroVLA 组内版）

> **定位调整**
>
> 这版计划不再以“广泛学习 VLA 并寻找任意 VLA 论文题目”为主线，而是以：
>
> **主流 VLA 基线认知 → 类脑计算基础 → NeuroVLA 深入理解与复现 → 围绕快慢系统、在线适应、记忆与低功耗控制形成研究问题**
>
> 为主线。
>
> 默认投入：每天约 3 小时，每周 6 天，第 7 天复盘/补缺。
>
> 原则：主流 VLA 学到“能看懂、能比较、能跑”即可；类脑部分要学到“能解释机制、能改模块、能做对照实验”。

---

# 0. 八周结束时的验收标准

八周结束时，至少完成以下 8 项：

- [ ] 能完整解释 `Image + Language + State → VLA → Action`。
- [ ] 能解释 RT-1、RT-2、OpenVLA、π0 的 Action 表示差异。
- [ ] 能解释为什么大 VLA 在高频机器人控制中会遇到延迟、抖动和时序问题。
- [ ] 能解释 LIF、SNN、surrogate gradient、STDP / R-STDP 的基本作用。
- [ ] 能完整画出 NeuroVLA 的 Cortex–Cerebellum–Spinal 三层结构。
- [ ] 能在 NeuroVLA 代码中定位：视觉/语言输入、意图提取、Cerebellar 模块、Spinal SNN、loss、action 输出、评测。
- [ ] 至少复现一组 NeuroVLA baseline / ablation，并能解释结果。
- [ ] 确定一个类脑具身研究问题，并完成 pilot、实验矩阵和论文骨架。

---

# 1. 总学习主线

```text
Robot Data / Action
        ↓
Behavior Cloning / Action Chunk / Diffusion
        ↓
RT-1 / RT-2 / OpenVLA
        ↓
π0 / π0.5
        ↓
理解主流 VLA 的瓶颈：
高延迟 / 高频反馈不足 / 时序记忆不足 / 在线适应困难 / 能耗
        ↓
SNN / LIF / Surrogate Gradient
        ↓
STDP / R-STDP / Event-driven Computing
        ↓
NeuroVLA
Cortex → Cerebellum → Spinal
        ↓
复现 + Ablation
        ↓
类脑具身研究问题
        ↓
Pilot → Paper Skeleton
```

---

# 2. π 系列怎么读

不再要求把 π 系列全部精读。

| 工作 | 建议 | 重点 |
|---|---|---|
| π0 | **精读** | VLM + Action Expert + Flow Matching + continuous action chunk |
| FAST | 快速读 | Action tokenization 的另一条路线 |
| π0.5 | **重点读** | open-world generalization、数据混合与泛化 |
| Real-Time Action Chunking | **精读** | 大模型延迟与实时动作执行矛盾 |
| π0.6 / RL 相关 | 选读 | 从 demonstration 走向真实执行经验学习 |
| Long/Short-Term Memory | **精读思想** | 长短期记忆、多时间尺度 |
| π0.7 | 后读 | 用作最新系统能力参照，不作为基础教材 |

判断标准：

> 读这些工作的目的不是掌握 PI 全部技术细节，而是回答：**普通 VLA 还缺什么，NeuroVLA 为什么需要另一种系统结构？**

---

# 3. Week 1：Robot Learning 最小基础

> 当前已完成的内容继续保留，不重复扩展。

## Day 1：Robot Data
- observation
- state / proprioception
- action
- episode
- trajectory
- demonstration
- timestep
- control frequency

## Day 2：Robot Action
- joint / end-effector / delta action
- `[T, D]`
- state 与 action 的区别
- feedback 为什么不需要单独叫 correction

## Day 3：Behavior Cloning
- `π(a_t | o_t)`
- expert demonstration
- distribution shift
- sequential error accumulation

## Day 4：Action Chunk
- chunk size
- horizon
- open-loop
- receding horizon
- temporal ensemble

## Day 5：Diffusion Policy
- multimodal action
- MSE 平均问题
- noise action trajectory
- denoising
- action trajectory

## Day 6：复盘

### Week 1 验收
不看笔记回答：
1. observation / state / action 区别；
2. trajectory / episode 区别；
3. BC 为什么会 distribution shift；
4. 为什么 action chunk 有用；
5. diffusion 为什么适合 action trajectory。

---

# 4. Week 2：VLM 怎么变成 VLA

## 本周目标

回答：

> **视觉、语言和机器人动作到底是怎么在一个模型里连起来的？**

## Day 1：RT-1

重点：
- 6 帧 image 输入；
- EfficientNet；
- language → USE → FiLM；
- `9×9×512`；
- TokenLearner；
- Transformer；
- discrete action token。

必须能解释：

```text
Image + Language
      ↓
language-conditioned visual representation
      ↓
Transformer
      ↓
Action
```

## Day 2：RT-2

重点：
- VLM 与 VLA 的关系；
- action tokenization；
- next-token prediction 为什么可以延伸到 action；
- web-scale knowledge 如何进入 robot control。

输出：

`notes/02_rt1_rt2.md`

## Day 3：OpenVLA

重点：
- Vision Encoder；
- Projector；
- LLM；
- Action Tokens；
- action quantization / decoding。

不要深入所有实现细节。

## Day 4：Open X-Embodiment

重点：
- embodiment 不同；
- action space 不同；
- camera / control frequency 不同；
- cross-embodiment 为什么难。

## Day 5：融合方式专题

用今天学习 RT-1 的方式比较：

```text
Early conditioning
FiLM

Feature interaction
Cross-attention

Late alignment
CLIP-style representation alignment
```

回答：

> 不同模态“什么时候融合”，为什么会影响下游 action？

## Day 6：Week 2 总图

画：

```text
Image
  +
Language
  +
State
  ↓
Representation / Fusion
  ↓
Action Generator
  ↓
Robot Action
```

### Week 2 验收
- [ ] 能解释 RT-1 的 FiLM。
- [ ] 能解释 TokenLearner。
- [ ] 能区分 CLIP alignment 与 FiLM conditioning。
- [ ] 能解释 RT-2 / OpenVLA action token。
- [ ] 能解释“融合层次”为什么是 VLA 的核心设计变量之一。

---

# 5. Week 3：现代 VLA 最小参照系

## 本周目标

不是追论文数量，而是回答：

> **主流 VLA 在 Action、实时控制、泛化和记忆上分别怎么做？**

## Day 1：π0

精读：
- Abstract；
- Figure 1 / Figure 2；
- Architecture；
- Training。

重点：

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

## Day 2：Flow Matching

只学到能读 π0：
- noise action；
- velocity field；
- continuous trajectory；
- inference。

回答：

> π0 为什么不走 OpenVLA 的逐维离散 action token 路线？

## Day 3：π0.5

重点：
- open-world generalization；
- 多来源数据；
- 高层语义能力和低层动作能力怎么共同保持。

## Day 4：Real-Time Action Chunking

重点只看一个问题：

> **VLA 推理慢，但机器人控制必须连续，怎么办？**

记录：
- latency；
- chunk；
- asynchronous / real-time execution；
- stale action 问题。

## Day 5：Memory + Online Learning

快速阅读：
- Long/Short-Term Memory；
- π0 后续 RL / online adaptation 工作。

只回答：
1. 为什么长任务不能只靠固定 context？
2. 为什么 deployment 后还需要 learning？
3. 普通 VLA 的更新方式为什么不适合毫秒级局部适应？

## Day 6：VLA 瓶颈地图

输出：

`notes/03_vla_bottlenecks.md`

固定画成：

```text
VLA 优势：
语义理解 / 泛化 / 多任务

VLA 局限：
├─ 高延迟
├─ 高频反馈处理弱
├─ action jitter
├─ 局部 reflex 缺失
├─ 长短期记忆
├─ 在线持续适应
└─ 能耗
```

### Week 3 验收

能够用 5 分钟解释：

> **为什么“把 VLM 做得更大”不等于机器人运动智能自然解决？**

---

# 6. Week 4：类脑计算最小基础

## 本周目标

回答：

> **SNN 为什么可能适合机器人，而不仅是另一种神经网络？**

## Day 1：LIF Neuron

理解：
- membrane potential；
- threshold；
- spike；
- leak；
- reset；
- 时间状态。

必须能解释：

> 为什么 SNN 天然带有时间状态，而普通无状态 MLP 没有？

## Day 2：Spike Representation

理解：
- rate coding；
- temporal coding；
- event-driven；
- sparse activation。

任务：
画一个连续 sensor signal → spike train 的 toy 图。

## Day 3：SNN Training

重点：
- spike 不可导问题；
- surrogate gradient；
- BPTT through time；
- ANN-to-SNN conversion 只了解概念。

不要推复杂数学。

## Day 4：STDP

重点：
- pre / post spike timing；
- synaptic plasticity；
- local learning；
- unsupervised adaptation。

## Day 5：R-STDP

重点：
- reward；
- eligibility；
- reward-modulated plasticity；
- 和 RL 的区别与联系。

必须回答：

> 为什么 R-STDP 对“机器人在线局部适应”有吸引力？

## Day 6：类脑控制总图

输出：

`notes/04_neuromorphic_basics.md`

画：

```text
High-level cognition
        ↓
slow

Adaptive feedback
        ↓
fast

Reflex / local control
        ↓
faster
```

并标注：
- information；
- control frequency；
- learning mechanism；
- memory timescale。

---

# 7. Week 5：NeuroVLA 精读 + 代码地图

## 本周目标

回答：

> **NeuroVLA 到底把传统 VLA 的哪几件事拆开了？为什么这么拆？**

NeuroVLA 的核心不是“把 VLA 换成 SNN”，而是建立：

```text
Cortex
  ↓
Cerebellum
  ↓
Spinal
```

三层不同时间尺度的控制结构。

## Day 1：整体架构

精读：
- Introduction；
- Figure 1；
- system overview。

重点：
- Cortex；
- Cerebellar Module；
- Spiking Spinal Module；
- high-frequency proprioception；
- force feedback；
- fast reflex。

输出一张完全自己画的 NeuroVLA 架构图。

## Day 2：Cortical Module

重点：
- VLM 负责什么；
- Q-Former / intention extraction；
- 为什么高层不直接输出所有高频动作；
- semantic intention 如何向下传递。

回答：

> Cortex 输出的是“动作细节”还是“运动意图”？

## Day 3：Cerebellar Module

重点：
- proprioception；
- force / wrench；
- adaptive filtering；
- gain modulation；
- smoothing；
- trajectory correction；
- temporal rhythm。

回答：

> Cerebellar Module 和普通 action smoothing filter 有什么根本差别？

## Day 4：Spiking Spinal Module

重点：
- LIF / SNN；
- event-driven sparsity；
- temporal integration；
- local fast loop；
- reflex；
- hardware mapping。

回答：

> Spinal SNN 为什么不仅是一个低功耗 action head？

## Day 5：NeuroVLA 代码地图

只定位：

```text
Dataset
Image / Language
Robot State
Force / Wrench
Cortical module
Q-Former / intention
Cerebellar module
Spinal SNN
Action output
Loss
Training loop
Evaluation
```

输出：

`notes/05_neurovla_code_map.md`

## Day 6：论文结果和 Ablation

重点看：
- No-Cerebellum；
- Single-step SNN；
- Multi-step SNN；
- jerk / acceleration；
- latency；
- energy；
- long-horizon / robustness；
- real robot reflex。

### Week 5 验收

不看论文画出：

```text
Vision + Language
       ↓
Cortex: semantic intention
       ↓
Cerebellum: high-frequency adaptive modulation
       ↓
Spinal SNN: temporal integration + fast execution/reflex
       ↓
Robot
```

并解释每一层为什么不能简单删掉。

---

# 8. Week 6：真正跑 NeuroVLA

## 本周目标

> **从“能读懂 NeuroVLA”变成“能改 NeuroVLA”。**

如果组内已有环境、数据、真机或 checkpoint，优先使用组内标准流程，不重复搭建另一套体系。

## Day 1：Baseline

跑通至少一个：
- LIBERO；
- 组内 simulation benchmark；
- 组内真机标准任务。

记录：
- success rate；
- inference latency；
- episode length；
- action smoothness。

## Day 2：No-Cerebellum Ablation

固定其他条件，只去掉 Cerebellar Module。

观察：
- success；
- jerk；
- collision recovery；
- temporal stability。

## Day 3：Single-step vs Multi-step SNN

回答：

> temporal state 到底有没有真实贡献？

## Day 4：Frequency / Latency Experiment

设计：
- cortex update frequency；
- cerebellar update frequency；
- spinal update frequency。

目标：

> 看不同层真正需要什么时间尺度。

## Day 5：Perturbation

至少选一个：
- visual noise / occlusion；
- force perturbation；
- collision；
- sensor delay；
- action delay。

## Day 6：Baseline 报告

输出：

`results/neurovla_baseline.md`

必须包含：
1. Model；
2. Task；
3. Input；
4. Frequency；
5. Metrics；
6. Main result；
7. Ablation；
8. Failure cases。

---

# 9. Week 7：形成真正的类脑研究问题

## 本周目标

不再做原计划中的通用 `Action-critical multimodal data valuation` 主线。

研究问题必须满足：

> **既有明确的机器人痛点，又能体现“类脑机制为什么必要”。**

## 方向 A：Adaptive Fast–Slow Routing —— 推荐优先

### 问题

目前 Cortex / Cerebellum / Spinal 的职责是预设的。

可以研究：

> **机器人能不能根据任务阶段、风险、预测不确定性或传感器变化，动态决定什么时候调用慢系统，什么时候主要依赖快系统？**

例如：

```text
free-space movement
→ 少调用 Cortex

contact / uncertainty ↑
→ Cerebellum / Spinal 高频工作

semantic goal changes
→ Cortex 更新
```

### 优点
- 与 NeuroVLA 体系直接连续；
- 不需要推翻现有 backbone；
- 能同时讨论 latency、energy、control quality；
- 很符合“多时间尺度脑机制”。

### 风险
- routing criterion 必须有明确物理意义；
- 不能只做一个普通 gating network。

推荐度：★★★★★

---

## 方向 B：Online Plasticity in Cerebellar / Spinal Loop

### 问题

> **低层回路能否在机器人执行过程中用局部反馈持续适应，而不是每次重新训练大 VLA？**

可探索：
- R-STDP；
- local error-based update；
- reward-modulated plasticity；
- force / proprioception driven adaptation。

### 优点
- 类脑属性最强；
- 与持续学习、在线学习直接连接；
- 可以研究 few-shot adaptation。

### 风险
- 学习稳定性难；
- 真机验证要求较高。

推荐度：★★★★★

---

## 方向 C：Multi-timescale Motor Memory

### 问题

> **短时运动状态、任务阶段和长期技能经验应该分别存在哪里？**

可以比较：
- cortical context；
- cerebellar temporal state；
- SNN membrane state；
- external memory。

### 优点
- 与长任务、遮挡、重复节律动作直接相关；
- 能形成清楚的 memory timescale 分析。

### 风险
- 容易和普通 memory model 混在一起；
- 必须强调 motor memory，而不是加一个 Transformer memory。

推荐度：★★★★☆

---

## 方向 D：Event / Force / Tactile Fast Reflex

### 问题

> **高频事件传感器是否应该绕过 VLA 大模型，直接进入 fast loop？**

### 优点
- 类脑和真实机器人结合紧；
- latency / safety 指标清楚。

### 风险
- 硬件依赖大；
- 数据采集成本高。

推荐度：★★★★☆

---

## Day 1–2：文献矩阵

不再泛搜全部 VLA。

文献分 4 类：
1. VLA latency / real-time；
2. neuromorphic robotics；
3. continual / online robot learning；
4. motor memory / hierarchical control。

## Day 3：Research Question

最多写 3 个。

格式：

> Existing VLA / NeuroVLA systems ______, but ______ remains unclear.

## Day 4：实验矩阵

固定：
- NeuroVLA baseline；
- 一个 conventional VLA baseline；
- 一个 ablation；
- proposed method。

## Day 5：Pilot

只做：
- 1 个任务；
- 1 个 baseline；
- 1 个 proposed change。

## Day 6：Go / No-Go

只有同时满足才继续：
- 有明显机制差异；
- 有量化指标；
- 不是简单调参；
- 能解释为什么是类脑问题；
- 组内现有工作没有已经解决同一问题。

---

# 10. Week 8：第一轮论文雏形

## Day 1：四句话

```text
Problem:
普通 VLA / 当前 NeuroVLA 的什么能力仍不足？

Biological Inspiration:
对应的生物机制是什么？

Method:
你增加了什么机制？

Result:
它改善了什么：success / latency / jerk / energy / adaptation / robustness？
```

## Day 2：Method Figure

必须明确画出：

```text
Slow Cortex
     ↓
Adaptive Mid-level
     ↓
Fast Spiking Loop
```

以及你新增模块在哪里。

## Day 3：Main Experiment

至少一张主表：

| Method | Success | Latency | Jerk | Robustness | Energy/Activity |
|---|---:|---:|---:|---:|---:|
| Baseline VLA | | | | | |
| NeuroVLA | | | | | |
| Proposed | | | | | |

指标按实际条件选，不要求全部都有。

## Day 4：Mechanism / Ablation

至少回答：

> 为什么有效？

而不是只说 success rate 提升。

## Day 5：Failure Analysis

分类：
- semantic planning；
- temporal phase；
- proprioceptive adaptation；
- reflex；
- SNN execution；
- sensor failure。

## Day 6：八周最终评估

项目至少包含：

```text
BRAIN_INSPIRED_VLA/
├── notes/
│   ├── 01_robot_data.md
│   ├── 02_rt1_rt2.md
│   ├── 03_vla_bottlenecks.md
│   ├── 04_neuromorphic_basics.md
│   └── 05_neurovla_code_map.md
├── literature/
│   └── brain_vla_matrix.md
├── experiments/
│   ├── baseline/
│   ├── neurovla/
│   └── proposed/
├── results/
│   ├── baseline.csv
│   ├── ablation.csv
│   └── failure_cases.md
└── paper/
    ├── research_question.md
    ├── method_figure.*
    ├── experiment_matrix.md
    └── outline.md
```

---

# 11. 论文阅读优先级

## A 类：必须真正理解

1. ACT  
2. Diffusion Policy  
3. RT-1  
4. RT-2  
5. OpenVLA  
6. π0  
7. NeuroVLA  

## B 类：围绕问题重点读

1. π0.5  
2. Real-Time Action Chunking  
3. Long/Short-Term Memory  
4. VLA online RL / continual adaptation 相关工作  
5. Neuromorphic robotics / SNN control 代表工作  

## C 类：知道即可

1. FAST  
2. SmolVLA  
3. OpenVLA-OFT  
4. π0.7  

---

# 12. 每篇论文统一问 10 个问题

```text
1. Problem：解决什么问题？
2. Input：输入是什么？
3. Output：输出是什么？
4. Action：Action 怎么表示？
5. Timescale：模型各模块以什么频率运行？
6. Feedback：有没有高频 state / force / tactile feedback？
7. Memory：时间信息存在哪里？
8. Learning：训练后还能不能继续适应？
9. Cost：latency / energy / computation 如何？
10. What I Can Change：对 NeuroVLA 下一步有什么可修改点？
```

这是后续读类脑 VLA 最重要的检查表。

---

# 13. 研究方向优先级调整

原来的主线：

```text
Action-critical multimodal data valuation
```

降为备选。

新的优先级：

| 方向 | 与组内 NeuroVLA 连续性 | 类脑属性 | 实验可解释性 | 推荐 |
|---|---:|---:|---:|---:|
| Adaptive Fast–Slow Routing | 高 | 高 | 高 | ★★★★★ |
| Online Cerebellar/Spinal Plasticity | 高 | 很高 | 高 | ★★★★★ |
| Multi-timescale Motor Memory | 高 | 高 | 中高 | ★★★★☆ |
| Event/Force/Tactile Reflex | 高 | 很高 | 高 | ★★★★☆ |
| Action-critical Data Valuation | 中 | 中 | 高 | ★★★ |
| 新通用 VLA backbone | 低 | 低 | 中 | ★★ |

---

# 14. 现在最重要的学习原则

不要把目标设成：

> “把 π 系列看完。”

而应该设成：

> **“我能用 π0、RTC、Memory 等主流工作作为参照，准确说出 NeuroVLA 为什么需要 Cortex–Cerebellum–Spinal 的多时间尺度结构，并能在这个结构上提出一个新的、可实验验证的问题。”**

这才是从“学习 VLA”进入“做类脑具身智能研究”的分界线。
