## Day 5：Diffusion Policy 总结

今天核心学了 6 件事：

1. **为什么需要 Diffusion Policy**
   同一个 observation 下，可能有多条合理 action trajectory。普通 MSE 容易把它们平均，得到不合理动作。

2. **Action 是多模态的**
   例如绕障碍物可以左绕，也可以右绕，两条路径都正确。Diffusion 的目标不是输出唯一答案，而是学习这些合理动作的分布。

3. **训练时怎么做**
   从 expert demonstration 中截取一个 action chunk，例如 `[16,7]`，给整个 chunk 加随机噪声。模型输入：

   ```text
   observation
   + noisy action chunk
   + diffusion timestep k
   ```

   模型学习预测加入的 noise，从而学会“当前 noisy action 应该往哪个合理动作方向修”。

4. **训练时不是 1→2→…→100 逐级加噪**
   而是随机选一个噪声等级 `k`，直接把 clean action 变成对应的 noisy action，再训练模型。

5. **推理时怎么做**
   当前真实 observation 下，从一个随机 action chunk 开始：

   ```text
   random action
   ↓
   k=100 去噪
   ↓
   k=99 再去噪
   ↓
   ...
   ↓
   k=0
   ↓
   合理 action chunk
   ```

   去噪全部完成后，机器人才执行其中前几步，再重新观察和规划。

6. **为什么最终能得到合理动作**
   不是因为随机噪声里藏着“正确路径”，而是因为模型通过大量 demonstration 学会了：

   > 从不同 noisy action 出发，应该往哪里修，才能靠近 expert action 的合理区域。

### 一句话记住 Day 5

> **Diffusion Policy 用“随机 action + 多步去噪”的方式，从 expert demonstration 中学习 action distribution，并在当前 observation 下生成其中一条合理的 action trajectory。**

你现在 Day 5 最重要的主线已经打通了：

```text
multimodal action
→ MSE averaging 问题
→ action chunk 加噪
→ 学习去噪方向
→ 从随机 noise 逐步生成
→ 得到一个合理 trajectory
```
