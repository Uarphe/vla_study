# VLA → 类脑具身智能：已完成基础 + 4 周研究启动计划

> **目标**
>
> 4 周内完成：  
> **主流 VLA 最小认知 → 类脑基础 → NeuroVLA 深入理解与代码复现 → 明确 research gap 并完成一个 pilot。**
>
> 默认投入：每天约 3 小时，每周 6 天，第 7 天复盘/补缺。
>
> 核心原则：**不再追求“把 VLA 论文看完”，只学到足以判断 NeuroVLA 的问题、机制和下一步研究空间。**

当前起点：Robot Data、Action、BC、Action Chunk、Diffusion Policy 已完成，因此新的 4 周从 **RT-1 / RT-2 / OpenVLA / π0** 开始。

---


# 已完成阶段：Robot Learning 最小基础（原 Week 1，不占后续 4 周）

这一部分保留在总计划里，用来表示已经完成的知识基础，**不再重复学习**。

| 主题 | 已学习内容 | 当前验收状态 |
|---|---|---|
| Robot Data | observation、state、action、episode、trajectory、demonstration、timestep、control frequency | ✅ 已完成 |
| Robot Action | joint position、joint velocity、delta joint、end-effector pose、delta end-effector pose、gripper；`[T,D]` | ✅ 已完成 |
| Behavior Cloning | expert demonstration、supervised learning、policy、distribution shift、`π(a_t \| o_t)` | ✅ 已完成 |
| Action Chunk | chunk size、horizon、open-loop、receding horizon、temporal ensemble | ✅ 已完成 |
| Diffusion Policy | multimodal action、MSE 平均问题、noise action trajectory、denoising、action trajectory | ✅ 已完成 |
| 综合理解 | Camera + Robot State + Language → Policy → Action Chunk → Robot | ✅ 已完成 |

已经具备的基础可以直接作为后续学习前提：

```text
Robot observation / state
        ↓
Policy
        ↓
Action / Action Chunk
        ↓
Robot execution
        ↓
New observation
```

后续 4 周不再重复这些内容，只在遇到 NeuroVLA、π0 或控制问题时回查。

---

# 0. 四周结束时的验收标准

四周后至少完成：

- [ ] 能完整解释 `Image + Language + State → VLA → Action`。
- [ ] 能说清 RT-1、RT-2、OpenVLA、π0 的核心差别。
- [ ] 能解释普通 VLA 的 5 个关键瓶颈：延迟、高频反馈、记忆、在线适应、能耗。
- [ ] 能解释 LIF、SNN、surrogate gradient、STDP / R-STDP。
- [ ] 能完整画出 NeuroVLA 的 Cortex–Cerebellum–Spinal 结构。
- [ ] 能在 NeuroVLA 代码中定位核心输入、模块、loss、action 与 evaluation。
- [ ] 至少复现一组 baseline / ablation。
- [ ] 建立一张 gap map，并筛出 1–2 个可做方向。
- [ ] 完成一个最小 pilot，决定 Go / No-Go。

---

# 1. 总路线

```text
[已完成]
Robot Data / Action / BC / Action Chunk / Diffusion
        ↓
[后续 Week 1]
RT-1 / RT-2 / OpenVLA / π0
        ↓
普通 VLA 的能力与瓶颈
        ↓
SNN / STDP / R-STDP
        ↓
NeuroVLA
Cortex → Cerebellum → Spinal
        ↓
代码 + Ablation
        ↓
Gap Map
        ↓
1 个 Pilot
```

---

# 2. Week 1：把普通 VLA 学到“够用”

## 本周目标

不是继续系统学 VLA，而是回答：

> **普通 VLA 怎么从视觉和语言生成动作？它还缺什么？**

### Day 1：RT-1
重点：
- 6 帧 image；
- language → USE → FiLM；
- EfficientNet；
- `9×9×512`；
- TokenLearner；
- Transformer；
- discrete action。

验收：

> 能不看资料完整讲一遍 RT-1 的输入到 action。

### Day 2：RT-2
重点：
- VLM → VLA；
- action token；
- next-token prediction；
- web knowledge → robot action。

验收：

> 能解释为什么机器人 action 可以被当作 token 预测。

### Day 3：OpenVLA
重点：
- Vision Encoder；
- Projector；
- LLM；
- Action Tokenization；
- action decode。

不读全部代码。

### Day 4：π0
重点：
- VLM；
- Action Expert；
- Flow Matching；
- Continuous Action Chunk。

只回答：

> 为什么 π0 不再走 OpenVLA 的逐维离散 token 路线？

### Day 5：普通 VLA 的瓶颈
建立：

`notes/vla_gap_map_v0.md`

至少写出：

| 能力 | 普通 VLA 怎么做 | 主要问题 |
|---|---|---|
| 语义理解 | VLM | 较强 |
| Action | token / flow / chunk | 高频精细控制仍难 |
| 实时性 | chunk / asynchronous | 大模型 latency |
| 高频反馈 | state / proprioception | 利用不足 |
| Memory | context / history | 长时间尺度有限 |
| Online learning | fine-tune / RL | 更新成本高 |
| 能耗 | dense model | 高 |

### Day 6：Week 1 复盘

必须能回答：

1. RT-1、OpenVLA、π0 最大区别是什么？
2. 为什么 action chunk 不能完全解决 latency？
3. 为什么“VLM 更大”不等于“机器人运动控制更好”？
4. NeuroVLA 可能是在补哪几个缺口？

---

# 3. Week 2：类脑基础 + NeuroVLA 读透

## 本周目标

回答：

> **NeuroVLA 为什么需要 Cortex–Cerebellum–Spinal，而不是再做一个更大的 VLA？**

### Day 1：LIF + SNN
重点：
- membrane potential；
- threshold；
- leak；
- reset；
- temporal state；
- sparse spike。

验收：

> 能解释 SNN 为什么天然具有时间状态。

### Day 2：Surrogate Gradient + STDP
重点：
- spike 不可导；
- surrogate gradient；
- pre/post timing；
- local plasticity。

### Day 3：R-STDP
重点：
- reward；
- eligibility；
- reward-modulated plasticity；
- online adaptation。

验收：

> 能解释 R-STDP 为什么对机器人在线局部适应有吸引力。

### Day 4：NeuroVLA 整体结构
重点：
- Cortex；
- Cerebellum；
- Spinal；
- 各层输入输出；
- 各层时间尺度；
- 哪一层负责语义、适应、反射。

输出：

`notes/neurovla_architecture.md`

### Day 5：NeuroVLA 模块细读
逐层回答：

```text
Cortex：
输入什么？
输出什么？
为什么不直接输出全部高频 action？

Cerebellum：
使用哪些高频反馈？
如何修正运动？

Spinal：
SNN 在哪里？
时间状态在哪里？
为什么适合快速 reflex？
```

### Day 6：NeuroVLA Ablation
重点只看：

- No-Cerebellum；
- Single-step vs Multi-step SNN；
- latency；
- jerk / smoothness；
- robustness；
- energy / spike activity；
- real robot perturbation。

输出：

`notes/neurovla_mechanism.md`

---

# 4. Week 3：代码 + 复现 + 真正找 gap

## 本周目标

从“理解论文”进入：

> **我知道 NeuroVLA 哪一块可以改。**

### Day 1：代码地图
只找：

```text
Dataset
Image / Language
Robot State
Force / Wrench
Cortex
Cerebellum
Spinal SNN
Action
Loss
Training
Evaluation
```

输出：

`notes/neurovla_code_map.md`

### Day 2：跑 Baseline
优先使用组内现成环境、checkpoint、benchmark。

记录：
- success rate；
- latency；
- episode length；
- action smoothness；
- failure case。

### Day 3：复现一个 Ablation
优先选一个：

1. No-Cerebellum；
2. Single-step vs Multi-step SNN；
3. 不同 control frequency。

目的不是追结果，而是确认：

> **我真的能修改 NeuroVLA。**

### Day 4：Gap Map v1
建立：

| 模块 | NeuroVLA 已解决 | 仍可能存在的问题 |
|---|---|---|
| Cortex | 高层语义 | 是否调用过于固定？ |
| Cerebellum | 高频适应 | 是否能在线学习？ |
| Spinal | 快速 SNN 控制 | 是否有长期可塑性？ |
| Memory | 多时间尺度状态 | 是否存在真正 motor memory？ |
| Routing | 固定层级 | 能否动态 fast–slow routing？ |
| Sensor | vision/state/force | event/tactile 是否可走 fast loop？ |

### Day 5：定向查文献
只围绕 2 个候选 gap 查，不再泛读 VLA。

优先：
- fast–slow routing；
- online plasticity；
- motor memory；
- neuromorphic reflex；
- continual robot learning。

### Day 6：选 1–2 个候选方向

候选优先级：

1. **Adaptive Fast–Slow Routing**
2. **Online Cerebellar / Spinal Plasticity**
3. Multi-timescale Motor Memory
4. Event / Force / Tactile Fast Reflex

每个方向必须写：

```text
Problem
Existing limitation
Biological inspiration
Proposed change
Metric
Expected failure mode
```

---

# 5. Week 4：Pilot + Go / No-Go

## 本周目标

不是写论文，而是回答：

> **这个 gap 值不值得继续做？**

### Day 1：确定一个 Research Question

最多一句：

> Existing NeuroVLA systems ______, but ______ remains unclear.

同时写 1 个核心 hypothesis。

### Day 2：实验矩阵

最小设计：

| Group | Method | Purpose |
|---|---|---|
| B1 | Conventional VLA | 普通基线 |
| B2 | NeuroVLA | 组内基线 |
| A1 | NeuroVLA ablation | 验证关键模块 |
| M1 | Proposed | 验证新机制 |

指标只选真正相关的：

- success；
- latency；
- jerk；
- recovery；
- robustness；
- adaptation speed；
- spike activity / energy proxy。

### Day 3–4：Pilot

只做：

```text
1 个 task
1 个 baseline
1 个 proposed change
1–2 个关键指标
```

不要全量训练。

### Day 5：结果分析

只回答：

1. 有没有明显信号？
2. 是 success 提升，还是 latency / stability / adaptation 提升？
3. 结果是否符合类脑机制预期？
4. 有没有反例？
5. 是否只是普通调参？

### Day 6：Go / No-Go

## Go
同时满足：
- 有机制差异；
- 有量化收益；
- 能解释为什么是类脑问题；
- 与已有工作有明确区别；
- 组内资源能支撑继续验证。

## No-Go
出现以下情况就换：
- 已有工作高度重复；
- 只提升很小但无法解释；
- 只是增加一个普通网络模块；
- 需要远超当前资源的新硬件；
- 机制和结果没有对应关系。

输出：

`paper/research_question.md`

包含：

```text
Problem
Gap
Biological Inspiration
Hypothesis
Method
Pilot Result
Next Experiment
```

---

# 6. 四周内论文阅读优先级

## 必须理解
1. RT-1
2. RT-2
3. OpenVLA
4. π0
5. NeuroVLA

## 围绕 gap 再读
1. π0.5
2. Real-Time Action Chunking
3. Memory
4. Online RL / Continual Learning
5. Neuromorphic Robotics

## 暂时不需要系统读
1. FAST
2. SmolVLA
3. OpenVLA-OFT
4. π0.7

---

# 7. 最终判断

四周结束时，不要求：

> “我已经掌握 VLA 和类脑全部理论。”

真正要求：

> **我知道普通 VLA 的瓶颈在哪里，知道 NeuroVLA 已经解决了什么，也能指出至少一个尚未解决、可以实验验证的类脑具身问题，并且已经跑过一个 pilot。**
