# VLA 8周学习与论文启动计划：飞书表格版

## 总目标

| 项目 | 内容 |
|---|---|
| 8周目标 | 从“不熟悉 VLA 体系”推进到“能读懂核心模型、跑通标准 VLA、修改训练/评测流程，并形成可继续写成论文的研究线” |
| 默认投入 | 每天约 3 小时，每周 6 天，第 7 天复盘和补缺 |
| 学习原则 | 不追求补完全部机器人学，只学习当前实验需要的内容 |
| 最终交付 | 能解释 VLA 输入-模型-动作输出链路；能读 robot dataset；能跑 LIBERO evaluation；能完成一次 fine-tuning 或小规模训练；能修改一个研究变量并完成对比实验；第 8 周形成论文问题、实验表格、方法草图和初步结果 |

## 8周总览

| 周次 | 阶段目标 | 核心问题 | 主要任务 | 必须产出 |
|---|---|---|---|---|
| Week 1 | Robot Learning 最小基础 | VLA 到底在预测什么？ | 学 robot data、action、BC、Action Chunk、Diffusion Policy | `notes/01_robot_data.md`；能解释 observation/state/action/episode/trajectory |
| Week 2 | VLM 如何变成 VLA | VLM 为什么能控制机器人？ | 读 RT-1、RT-2、Open X-Embodiment、OpenVLA；读 OpenVLA 代码 | `notes/02_openvla_code_map.md`；能定位 image/language/action/loss/inference |
| Week 3 | 现代 VLA 路线 | Action token 和 continuous action 有什么区别？ | 学 π0、Flow Matching、OpenVLA vs π0、FAST、SmolVLA/OpenVLA-OFT | `notes/03_openvla_vs_pi0.md`；画出 VLA action generation 总图 |
| Week 4 | 真正跑通 VLA | Dataset → Policy → Environment → Action → Success Rate 是否能闭环？ | 装环境、读 LIBERO 数据、跑 pretrained policy、分析失败、做一个小 ablation | `experiments/baseline/`；success rate；failure cases；baseline 报告 |
| Week 5 | 确定论文问题 | 我是否正在研究一个具体 VLA 问题？ | 文献矩阵、Research Question、实验矩阵、pilot、Go/No-Go | `literature/paper_matrix.md`；RQ；pilot 结果 |
| Week 6 | 第一轮完整实验 | 方法是否有效？ | Baseline、主方法、重复实验、主结果表 | `results/main_results.csv`；第一张正式主表 |
| Week 7 | Ablation + Mechanism | 为什么有效？ | Ablation、Sensitivity、Failure Analysis、Visualization | `results/ablation.csv`；failure analysis；Figures 1-4 草图 |
| Week 8 | 形成论文雏形 | 能否讲清楚论文故事？ | Problem/Observation/Method/Result；Introduction；Methods；Experiments | `paper/outline.md`；`intro.md`；`method.md`；`experiments.md` |

## Week 1：Robot Learning 最小基础

| 天 | 主题 | 学习重点 | 实际任务 | 当日产出 / 验收 |
|---|---|---|---|---|
| Day 1 | Robot Data 基础 | observation、state、action、episode、trajectory、demonstration、timestep、control frequency | 找一个 LeRobot dataset，打印 `sample.keys()`，确认能找到 image/state/action/task 或 instruction | 新建 `notes/01_robot_data.md`，回答 observation/state/action/episode/trajectory 的含义 |
| Day 2 | 机器人 Action | joint position、joint velocity、delta joint、end-effector pose、delta end-effector pose、gripper；理解 `[x,y,z,rx,ry,rz,gripper]` | 从 dataset 取一个 episode，画 action 每个维度随 timestep 的曲线 | 能解释 `[T,D]` action tensor 中 T、D、每一行、相邻 action 相关性的含义 |
| Day 3 | Behavior Cloning | expert demonstration、supervised learning、policy、distribution shift；理解 `π(a_t \| o_t)` | 阅读 ACT 的 Introduction、Action Chunking、Figure 2、Experiments | 用不超过 300 字解释：为什么机器人模仿学习不能简单理解成普通图像分类 |
| Day 4 | Action Chunk | chunk size、horizon、open-loop execution、receding horizon、temporal ensemble | 写一个 toy 示例，把单步 action `[B,7]` 改成 action chunk `[B,16,7]` | 能解释一次预测多个 action 的意义 |
| Day 5 | Diffusion Policy | observation 作为 condition；noise action trajectory；denoising；action trajectory | 阅读 Diffusion Policy，重点理解 multimodal action、MSE 平均动作问题、为什么 diffusion 适合生成 action trajectory | 回答 3 个问题：action 为什么 multimodal；MSE 为什么会平均；diffusion 为什么适合 action trajectory |
| Day 6 | Week 1 复盘 | 独立画出 Camera + Robot State + Language → Policy → Action Chunk → Robot | 不看笔记回答 BC、Action Chunk、state/action、Diffusion vs regression | 不清楚的点放到 Day 7 补齐 |

## Week 2：VLM 如何变成 VLA

| 天 | 主题 | 学习重点 | 实际任务 | 当日产出 / 验收 |
|---|---|---|---|---|
| Day 1 | RT-1 | 输入、输出、action 表示、Transformer 的作用 | 阅读 RT-1 关键部分 | 画一张 RT-1 结构图 |
| Day 2 | RT-2 | VLM：Image + Language → Text Token；VLA：Image + Language → Action Token | 阅读 RT-2 | 回答：为什么 action token 化以后可以沿用 next-token prediction |
| Day 3 | Open X-Embodiment | 不同机器人、action space、camera、control frequency、tasks | 阅读 Open X-Embodiment，重点看数据异构性 | 回答：为什么 cross-embodiment robot learning 比普通多任务学习更难 |
| Day 4 | OpenVLA | Vision Encoder → Projector → LLM → Action Tokens；Action Tokenization | 阅读 OpenVLA | 能解释连续动作如何离散成 action token |
| Day 5 | OpenVLA 代码 | image、language tokenize、action tokenize、GT action、loss、inference decode | 只找关键路径，不通读全部代码 | 新建 `notes/02_openvla_code_map.md` |
| Day 6 | Week 2 复盘 | VLM → 加入 action representation → VLA | 自己解释 VLM 和 VLA 的差别 | 能解释 action token、OpenVLA 输入输出、cross-embodiment、action 编解码位置 |

## Week 3：现代 VLA 路线

| 天 | 主题 | 学习重点 | 实际任务 | 当日产出 / 验收 |
|---|---|---|---|---|
| Day 1 | π0 整体结构 | VLM semantic representation、Action Expert、Flow Matching、Continuous Action Chunk | 阅读 π0 的 Abstract、Figure 1/2、Architecture、Training | 建立 π0 整体结构图 |
| Day 2 | Flow Matching | noise action → velocity field → real action | 只学到能理解 π0 即可，不推完整数学 | 回答：π0 为什么不需要像 OpenVLA 一样离散 action token |
| Day 3 | OpenVLA vs π0 | discrete token/autoregressive vs continuous action/flow matching | 做对比表 | 新建 `notes/03_openvla_vs_pi0.md` |
| Day 4 | FAST | action sequence → frequency representation → compressed tokens → autoregressive generation | 阅读 FAST | 回答：FAST 想解决 OpenVLA 式 action tokenization 的什么问题 |
| Day 5 | SmolVLA / OpenVLA-OFT | SmolVLA：更小、Action Expert、asynchronous inference、训练参数；OpenVLA-OFT：action chunk、continuous action head、parallel decoding、proprioception | 阅读两篇论文的架构和实验 | 能说出两者和 OpenVLA/π0 的关系 |
| Day 6 | Week 3 总图 | VLA Action Generation 两条路线 | 画出 Discrete token 路线和 Continuous 路线 | 能讲清 OpenVLA vs π0、Flow Matching、FAST、Action Expert、Action Chunk vs Action Token |

## Week 4：真正跑通 VLA

| 天 | 主题 | 目标 | 实际任务 | 当日产出 / 验收 |
|---|---|---|---|---|
| Day 1 | 环境 | 确认 CUDA、PyTorch、LeRobot、MuJoCo/LIBERO 正常 | `conda activate vla`；`lerobot-info` | 建立 `experiments/baseline/` |
| Day 2 | 读取 LIBERO 数据 | 读一个 episode | 读取 image、state、action、language instruction | 保存 `episode_example.png`、`action_curve.png` |
| Day 3 | 跑 pretrained policy | 优先 π0 / SmolVLA；LIBERO-Object；10 episodes | 记录 success/failure、episode length、inference time | 得到第一组 success rate |
| Day 4 | 检查失败案例 | 分析 5 个成功和 5 个失败 | 判断失败类型：目标看错、抓取位置错误、动作不稳定、执行慢、语言理解错误、长时序误差 | 建立 `results/failure_cases.md` |
| Day 5 | 修改一个简单变量 | 证明自己能改实验，不只是运行命令 | 只改一个变量：action chunk size / camera views / proprioception on-off | 得到一个小 ablation |
| Day 6 | Baseline 报告 | 2 页以内总结 | 写 Model、Dataset、Input、Output、Evaluation、Success Rate、Failure Cases、Ablation | Week 4 没完成则禁止进入论文创新阶段 |

## Week 5：确定论文问题

| 天 | 主题 | 目标 | 实际任务 | 当日产出 / 验收 |
|---|---|---|---|---|
| Day 1-2 | 文献矩阵 | 确认是否已有高度相似工作 | 建立至少 15 篇近年论文矩阵 | `literature/paper_matrix.md` 或 `paper_matrix.xlsx` |
| Day 3 | Research Question | 用一句话写清问题 | 格式：Existing VLA methods ___, but ___ remains unclear. 再写 RQ1/RQ2/RQ3 | 最多 3 个 RQ |
| Day 4 | 实验矩阵 | 明确 baseline、method、model、benchmark、metric | 设计 Full、Random、Proposed、跨模型验证等组别 | 实验表格初稿 |
| Day 5 | Pilot | 只做 1 个 task suite、1 个 baseline、1 个 proposed method | 看有没有明显信号 | pilot 结果 |
| Day 6 | Go / No-Go | 决定是否继续方向 | 判断可运行性、结果是否负面、机制是否可解释、与已有论文区别 | Go 则正式定题；No-Go 则立刻换方向 |

## Week 6：第一轮完整实验

| 天 | 主题 | 目标 | 实际任务 | 当日产出 / 验收 |
|---|---|---|---|---|
| Day 1 | Baseline | 至少完成 Base model、Random baseline、已有方法 baseline | 固定 model、dataset split、steps、batch size、seed、eval episodes | baseline 结果 |
| Day 2-3 | 主方法 | 完成 Proposed Method | 至少跑 LIBERO-Spatial 和 LIBERO-Object；算力足够再加 Goal、LIBERO-10 | proposed 结果 |
| Day 4 | 重复实验 | 关键结果至少多 seed / 多次 evaluation | 记录 mean、std、episodes | 稳定性结果 |
| Day 5 | 第一张主表 | 形成正式主结果表 | 表格：Method、Spatial、Object、Goal、Long、Avg | `results/main_results.csv` |
| Day 6 | 结果判断 | 回答是否有效、在哪些任务有效/无效、为什么、有无反例 | 分析结果而不是只报平均提升 | 明确下一轮实验方向 |

## Week 7：Ablation + Mechanism

| 天 | 主题 | 目标 | 实际任务 | 当日产出 / 验收 |
|---|---|---|---|---|
| Day 1-2 | Ablation | 回答哪些模块有用 | 比如 without visual score / language score / action score / full | ablation 表 |
| Day 3 | Sensitivity | 回答数据比例或参数变化是否稳定 | 例如 25/50/75/100% 或 chunk=4/8/16 | sensitivity 表 |
| Day 4 | Failure Analysis | 分类失败原因 | Perception、Language grounding、Motion、Grasp、Long-horizon | failure cases 更新 |
| Day 5 | Visualization | 准备论文图 | Figure 1 Method；Figure 2 Main result；Figure 3 Ablation；Figure 4 Failure/Qualitative | figures 草图 |
| Day 6 | 重写主线 | 根据结果更新 RQ、Hypothesis、Main Finding | 明确为什么有效 | 方法机制解释 |

## Week 8：形成论文雏形

| 天 | 主题 | 目标 | 实际任务 | 当日产出 / 验收 |
|---|---|---|---|---|
| Day 1 | 确定故事 | 用四句话讲清论文 | Problem、Observation、Method、Result | `paper/outline.md` |
| Day 2 | Introduction 骨架 | 四段式 intro | P1 VLA 重要性；P2 当前问题；P3 现有方法不足；P4 本文方法和贡献 | `paper/intro.md` |
| Day 3 | Methods | 画出 Input → VLA → 你的模块 → Action | 写 Problem Formulation、Method Overview、核心公式、Training/Inference | `paper/method.md` |
| Day 4 | Experiments | 写完整实验设置 | Datasets、Benchmarks、Baselines、Implementation、Metrics、Main Results | `paper/experiments.md` |
| Day 5 | Results + Analysis | 每张表对应一个问题 | 主结果、机制分析、数据少时是否有效、跨模型是否有效 | results 草稿 |
| Day 6 | 最终评估 | 检查项目结构是否完整 | 对照最终文件清单 | 形成可继续扩展到投稿状态的 paper skeleton |

## 论文方向候选

| 方向 | 核心问题 | 优点 | 缺点 | 推荐度 |
|---|---|---|---|---|
| 普通 trajectory selection | 哪条 trajectory 更好 | 上手快、算力低 | 2026 撞题风险很高、机制深度弱 | 低 |
| Action-critical multimodal data valuation | 哪些 observation/segment 中的视觉、语言、状态信息真正改变 action prediction | 区别于普通 trajectory selection；机制分析强；可做跨任务/跨模型比较 | 需要认真查 2025-2026 最新工作；flow policy scoring 要设计严谨 | 高 |
| Adaptive Action Chunk | 不同动作阶段是否需要不同 chunk size | 问题直接、解释性强、action 侧价值高 | 代码改动更大，inference/evaluation 要严谨 | 高 |
| VLA Evaluation Robustness | 扰动后是否仍然有效 | 算力低、易快速形成大量结果 | 单纯 benchmark 测试创新不足，需要新诊断指标或系统性发现 | 中 |
| 新 VLA backbone | 设计新的 VLA 主干 | 机制深度高 | 算力高、周期长、风险大 | 低 |

## 推荐主线

| 项目 | 内容 |
|---|---|
| 推荐方向 | Action-critical multimodal data valuation for VLA fine-tuning |
| 核心不是 | 简单挑“高质量数据” |
| 核心是 | 研究视觉、语言、机器人状态在一个 trajectory 的不同阶段，对 action prediction 到底贡献多少 |
| 可形成结果 | Main Result、Data Efficiency、Modality Analysis、Task Analysis、Cross-model Validation、Failure Analysis |
| 注意 | Week 5 完成最新文献检查前，不锁死题目名称 |

## 每篇论文笔记模板

| 问题 | 要回答的内容 |
|---|---|
| 1. Problem | 它解决什么问题？ |
| 2. Previous Limitation | 以前为什么做不好？ |
| 3. Input | 模型输入什么？ |
| 4. Output | 模型输出什么？ |
| 5. Action Representation | Action 怎么表示？ |
| 6. Method | 核心方法是什么？ |
| 7. Experiment | 用什么数据、benchmark、metric？ |
| 8. What I Can Use | 对我的实验有什么直接价值？ |

## 代码阅读检查表

| 检查项 | 是否找到 | 文件/函数/备注 |
|---|---|---|
| Dataset loader |  |  |
| Image preprocessing |  |  |
| Language tokenizer |  |  |
| Robot state preprocessing |  |  |
| Action normalization |  |  |
| Action representation |  |  |
| VLM backbone |  |  |
| Action head / action expert |  |  |
| Loss |  |  |
| Action decoding |  |  |
| Action chunk |  |  |
| Evaluation loop |  |  |

## 实验记录模板

| 字段 | 内容 |
|---|---|
| Experiment ID |  |
| Hypothesis | 为什么做？ |
| Change | 只改了什么？ |
| Fixed | 哪些条件完全不变？ |
| Model |  |
| Dataset |  |
| Seed |  |
| Steps |  |
| Batch Size |  |
| Chunk Size |  |
| Success Rate |  |
| Mean |  |
| Std |  |
| Observation | 发生了什么？ |
| Conclusion | 支持还是否定 hypothesis？ |
| Next | 下一步唯一需要做什么？ |

## 最短执行版

| 周次 | 最忙时只保留这些任务 |
|---|---|
| Week 1 | ACT；Diffusion Policy；读 Robot Dataset |
| Week 2 | RT-2；OpenVLA；读 OpenVLA code |
| Week 3 | π0；FAST；SmolVLA |
| Week 4 | LeRobot；LIBERO；跑通 pretrained VLA |
| Week 5 | 最新文献矩阵；确定 Research Question；完成 pilot |
| Week 6 | Baseline；Main Experiment；第一张主表 |
| Week 7 | Ablation；Mechanism；Failure Analysis |
| Week 8 | Figure；Paper Outline；Introduction；Methods；Experiments |

## 禁止事项

| 不做 | 原因 |
|---|---|
| 从头学完整机器人运动学教材 | 当前目标是 VLA 实验闭环 |
| 从头学完整控制理论 | 只补实验需要的部分 |
| 为了 VLA 先系统学习 RL | 容易偏离主线 |
| 一开始训练 foundation model | 算力和周期不合适 |
| 同时跑 5 个 VLA | 分散精力，难以形成研究结论 |
| 同时研究数据、结构、RL、world model | 问题过宽 |
| 因为一篇新论文出现就立刻换方向 | 容易失去连续实验积累 |
| 只看 loss，不做 robot rollout | VLA 必须闭环评测 |
| 只报告单次 success rate | 不稳定，缺乏说服力 |
| 把 benchmark 提升自动解释成 generalization 提升 | 需要扰动测试或机制分析支持 |
| 没有 baseline 就写“创新方法” | 论文说服力不足 |

