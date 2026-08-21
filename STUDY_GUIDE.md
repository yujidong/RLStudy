# 学习指南与自测题（STUDY_GUIDE）

> **怎么用这份指南**：每学完一章，合上 notebook，凭记忆回答该章的自测题。
> 能答上来 = 过，去下一章；答不上来 = 回到 notebook 里对应的 `##` 小节重看。
> 这是**检索练习（retrieval practice）**——比"再看一遍"有效得多。
>
> 自测题的答案都在可折叠块里（点开前先自己说一遍）。
>
> **关于练习**：Ch01-05 的练习配 `solutions/` 参考答案；Ch06-13 的练习在各章 notebook
> 末尾（带可折叠提示 + 预期结果，开放练习无标准答案）；Ch15-18 以自测题为主。
> 建议顺序：先做题、再看提示、对预期结果，最后做本指南的自测题。

## 学习路径与时间预估

| Phase | 章节 | 预计投入 |
|---|---|---|
| 1 经典基础 | Ch00 → 05（+ 05b PyTorch 速成） | 10-15 小时 |
| 2 策略梯度 + PPO | Ch06 → 09 | 12-18 小时 |
| 3 LLM RLHF + GRPO | Ch10 → 15 | 15-20 小时 |
| 4 研究前沿 | Ch16 → 18 | 8-12 小时 |
| 5 Agentic RL | Ch19 → 20 | 4-5 小时 |

- **零基础**：按顺序走，别跳。
- **有 RL 基础赶时间**：Fast-track `Ch00 → 01 → 05 → (05b) → 07 → 09 → 13`，约 20 小时直达 GRPO。
- **每章的自测不过关就别前进**——后面的章节都默认你掌握了前面的内容。

---

## Phase 1

### Ch00 环境搭建 + RL 全景

1. 用自己的话说清 agent-environment loop：每个时刻依次发生哪三件事？
<details><summary>答案</summary>

① agent 按策略 π(a|s) 选动作 a_t；② environment 转移到 s_{t+1} 并给出标量奖励 r_{t+1}；③ agent 的目标是最大化期望累计奖励 E[Σ γ^t r_{t+1}]，而不是即时的 r。
</details>

2. 监督学习和强化学习的反馈有什么本质区别？为什么这个区别引出了 credit assignment 问题？
<details><summary>答案</summary>

监督学习每个输入都有标准答案（即时、确定）；RL 只有延迟、稀疏的奖励信号。一局棋赢了 +1，但**哪几步导致了胜利**信号里没说——把最终奖励分配回各个动作，就是 credit assignment，也是 Ch08 GAE 要解决的核心问题。
</details>

3. ClickWorld 演示里，贪心策略 18 步到目标但净奖励是 0。为什么"能到目标"不等于"好策略"？
<details><summary>答案</summary>

贪心路线踩了陷阱（-1）再到目标（+1），净收益 0；绕开陷阱的路线虽然更長，净收益 +1 更高。RL 优化的目标是**累计奖励**，不是"到达"本身——奖励函数怎么设计，agent 就优化什么。
</details>

### Ch01 多臂老虎机

1. ε-greedy 中 ε=0 和 ε=1 分别是什么行为？ε 固定不变有什么问题？
<details><summary>答案</summary>

ε=0 纯利用（可能永远锁死在次优臂）；ε=1 纯探索（随机乱拉）。固定 ε 即使已经确知最优臂也永远以 ε 概率乱拉——所以实践中 ε 要随时间衰减，或改用乐观初始化 / UCB / Thompson 这类"不确定性驱动"的方法。
</details>

2. UCB1 的选臂公式 `Q(a) + c·√(ln t / N(a))` 里，两项各代表什么？N(a) 变大时会发生什么？
<details><summary>答案</summary>

第一项是利用（当前估计的价值），第二项是探索奖励（不确定性）：拉得越少的臂（N(a) 小）bonus 越大，越值得试。N(a)→∞ 时 bonus→0，算法收敛到纯利用——探索量自动衰减，这正是 ε-greedy 做不到的。
</details>

3. 为什么"乐观初始化"（Q 初值设很高）能鼓励探索？
<details><summary>答案</summary>

每个臂的 Q 一旦被拉过就会被拉低到真实值附近；没拉过的臂 Q 还停在高位，greedy 会优先去试它们——"失望"本身驱动了系统的遍历。
</details>

4. 非平稳环境下，为什么样本平均 Q(a) 会失效、步长 α 的 EWMA 反而合适？
<details><summary>答案</summary>

样本平均对所有历史等权重，环境一变旧数据就变成毒药；EWMA `Q ← Q + α(r − Q)` 以 (1−α) 的速率指数遗忘旧数据，α 控制有效记忆长度，能追踪漂移。
</details>

### Ch02 MDP + 贝尔曼方程

1. 写出贝尔曼期望方程（V 的版本），并解释右边那两个期望分别在平均什么。
<details><summary>答案</summary>

V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ r(s,a,s') + γ V^π(s') ]。外层对**策略的动作选择**平均，内层对**环境的随机转移**平均。"当前的值 = 即时奖励 + 折扣 × 后继状态的值"这个递归是所有 RL 算法的根源。
</details>

2. γ=0 和 γ→1 各代表什么极端？为什么实践中常用 0.9~0.99？
<details><summary>答案</summary>

γ=0 目光短浅（只看下一步即时奖励）；γ→1 完全长视（回报等于整条轨迹总和，方差大、可能发散）。中间值既让远处奖励仍然重要，又保证无限长轨迹的回报有限（几何级数收敛）、方差可控。
</details>

3. Ch02 用蒙特卡洛采样验证了解析解 V。这种"解析解 vs 采样估计"的对照为什么重要？
<details><summary>答案</summary>

DP 的解是精确的，MC 是从真实交互估计的——两者在容差内一致，说明"贝尔曼方程确实描述了环境"。后面所有无法解析求解的算法（TD、Q-learning……）都靠采样，这个验证建立了"采样可以信任"的信心。
</details>

4. Q^π 和 V^π 的关系式是什么？为什么"知道 Q"就能直接选动作，"知道 V"还不能？
<details><summary>答案</summary>

Q^π(s,a) = Σ_{s'} P(s'|s,a)[r + γ V^π(s')]。V 只告诉你状态好不好，比较动作还需要转移模型 P；Q 直接给每个 (s,a) 打分，π(s)=argmax_a Q(s,a) 不再需要模型——这就是 Ch05 无模型控制的基础。
</details>

### Ch03 动态规划

1. 策略迭代的两步分别是什么？为什么交替执行会收敛到最优？
<details><summary>答案</summary>

策略评估（算当前策略的 V^π）+ 策略改进（对每个状态贪心地取 argmax_a Q）。策略改进定理保证每次改进后的策略不劣于旧的；策略数有限，所以有限步内收敛到最优。
</details>

2. 值迭代和策略迭代的区别是什么？各自的收敛条件？
<details><summary>答案</summary>

值迭代把"评估"压缩成一次贝尔曼最优回扫（V←max_a Σ P[r+γV']），每步都做 max；策略迭代做完整评估再改进，通常少数几轮就收敛但每轮更贵。两者都要求 γ<1（压缩映射，误差以 γ 每轮几何衰减）。
</details>

3. DP 有什么致命的实际限制？（提示：两个）
<details><summary>答案</summary>

① 需要**完整的模型** P(s'|s,a)、r —— 真实世界通常未知；② 状态数**指数爆炸**（扫一遍所有状态 × 动作）。Ch04 的 TD 学习解决 ①（从样本学），Ch06 的函数逼近解决 ②。
</details>

### Ch04 TD 学习

1. 写出 TD(0) 更新式，并指出哪部分是 TD target、哪部分是 TD error。
<details><summary>答案</summary>

V(s_t) ← V(s_t) + α[ r_{t+1} + γV(s_{t+1}) − V(s_t) ]。方括号整体是 TD error δ_t；r+γV(s_{t+1}) 是 TD target（对真实回报的自举估计）。MC 的 target 是整条轨迹的实际回报。
</details>

2. MC 和 TD(0) 的 bias/variance 权衡分别怎么说？
<details><summary>答案</summary>

MC target 用真实最终回报：无偏但方差高（整条轨迹的随机性都算进去）。TD target 用 V 的自举估计：方差低（只含一步转移的随机性）但**有偏**（V 还不准时 target 就不准）。n-step TD / TD(λ) 在两者之间插值。
</details>

3. λ=0 和 λ=1 的 TD(λ) 分别退化成什么？
<details><summary>答案</summary>

λ=0 → TD(0)（一步自举）；λ=1 → 蒙特卡洛（把整条轨迹的误差无折扣地传播回来）。中间的 λ 兼顾偏差与方差——Ch08 的 GAE 用的是完全相同的思想，只是把 V 换成了 advantage。
</details>

4. eligibility trace 的直觉是什么？它和"每步只更新一个状态"的 TD(0) 差在哪？
<details><summary>答案</summary>

给每个状态维护一个"新鲜度"痕迹：被访问就 +1，每步 ×γλ 衰减。出现 TD error 时**所有近期访问过的状态**按痕迹比例一起更新——一次 credit assignment 到多个历史状态，不用等到回合结束（在线、每步更新）。
</details>

### Ch05 Q-learning / SARSA

1. 写出 Q-learning 和 SARSA 的更新式，圈出唯一的区别。
<details><summary>答案</summary>

Q: Q(s,a) ← Q(s,a) + α[ r + γ·**max_a'** Q(s',a') − Q(s,a) ]；SARSA: … γ·**Q(s', a')** …（a' 是**实际执行的下一个动作**）。Q-learning 的 target 与行为策略无关（off-policy），SARSA 用实际采样的动作（on-policy）。
</details>

2. CliffWalk 里 SARSA 学到的路线和 Q-learning 有什么不同？各自为什么？
<details><summary>答案</summary>

SARSA 学到离悬崖远的保守路线（它评估的是"带探索的自己"，探索性落崖的代价被计入）；Q-learning 学到贴崖捷径（它评估贪心策略的最优下一步，不管自己探索时会不会掉下去）。同一定义环境，行为策略的差别导致不同均衡。
</details>

3. 什么是 maximization bias？Double Q-learning 怎么解决？
<details><summary>答案</summary>

max_a Q(s',a) 系统性**高估**：噪声下"最大值"偏向被高估的那个臂。Double Q 把动作选择和评估拆到两组独立估计：用 Q₁ 选 a*、用 Q₂ 评估 Q₂(s',a*)，高估不再自我强化。（Ch06 的 Double DQN 是同一思想的神经网络版。）
</details>

4. 用一句话说清 on-policy / off-policy，并各举一例（本课程内）。
<details><summary>答案</summary>

on-policy：只能用**当前策略**采的数据训练（SARSA、策略梯度、PPO）；off-policy：可以用旧策略/别的策略采的数据训练（Q-learning、DQN）。这个区别决定了 PPO 为什么要 importance ratio、GRPO 为什么每次 rollout 后只更新几个 epoch。
</details>

### Ch05b PyTorch 速成

1. 默写训练五步循环，并说明漏掉第①步会发生什么。
<details><summary>答案</summary>

zero_grad → forward → loss → backward → step。漏掉 zero_grad：PyTorch 梯度默认**累加**，这次更新会混入之前所有 batch 的旧梯度，loss 乱跳不收敛。
</details>

2. `gather` 在 DQN 更新里干什么？`torch.no_grad()` 包住 target 计算又是为什么？
<details><summary>答案</summary>

gather 从网络输出的 [B, n_actions] 里挑出**实际执行动作**的 Q 值 [B,1]。target 是监督信号，不该对它求梯度（半梯度方法），no_grad 同时省内存防误更新。
</details>

3. `model.eval()` 忘了切回 `model.train()` 会怎样？
<details><summary>答案</summary>

dropout / batchnorm 停在推理行为，后续训练用的"网络"和你以为的不一样——损失曲线异常、评估随机跳变。规则：采样/评估前 eval()，**用完立刻 train()** 回来。
</details>

---

## Phase 2

### Ch06 DQN + 函数逼近

1. 表格 Q-learning → DQN，哪三样东西是新增的？各自解决什么问题？
<details><summary>答案</summary>

① 神经网络逼近 Q(s,a;θ)——解决状态爆炸/连续状态；② experience replay——打破样本相关性 + 一份数据多次利用；③ target network——固定几步的 TD target，防止"追自己尾巴"的震荡发散。
</details>

2. 什么是 deadly triad？为什么 naive DQN 会发散？
<details><summary>答案</summary>

函数逼近 + bootstrapping（自举）+ off-policy 三者同时出现时训练不稳定：网络一更新，所有状态的 Q 一起变，target 也跟着变，误差自我放大。replay + target network 就是为了拆掉这个反馈环的两个部件。
</details>

3. 半梯度（semi-gradient）和全梯度差在哪？TD target 为什么要 `no_grad`？
<details><summary>答案</summary>

半梯度只对 Q(s,a;θ) 求梯度，把 target r+γmax Q(s';θ⁻) 当常数（θ⁻ 是冻结的 target 网络参数）。若对 target 也求梯度（全梯度），优化目标每步都在漂移，实际是解一个不动的方程组而非最小化固定 loss。
</details>

4. Double DQN 改了哪一行？Dueling 网络把 Q 拆成了什么？
<details><summary>答案</summary>

Double：a* = argmax 用 online 网络选、Q(s',a*) 用 target 网络评（拆开选择与评估，治 maximization bias）。Dueling：Q(s,a) = V(s) + A(s,a) − mean(A(s,·))，状态价值和动作优势分开建模——很多状态下 V 好学但动作间差异小，拆开效率更高。
</details>

### Ch07 策略梯度

1. 写出策略梯度定理（轨迹形式），并解释 ∇θ log π 为什么是"方向"、Q 为什么是"权重"。
<details><summary>答案</summary>

∇J(θ) = E_τ[ Σ_t ∇θ log π(a_t|s_t) · Q(τ) ]。增大 log π 就是增大该动作的概率；乘上 Q>0 强化它、Q<0 抑制它、|Q| 大的动作动得多。整体效果 = 沿"提高高回报动作概率"的方向调参。
</details>

2. 为什么可以直接对 ∇θ log π 求梯度而不管转移概率 P？
<details><summary>答案</summary>

score function 技巧：∇ log p(τ) = ∇(log π 部分 + log P 部分)，环境的 P(s'|s,a) 不含 θ，求导为 0——所以策略梯度**不需要环境模型**，这正是它能用于真实环境/LLM 的原因。
</details>

3. baseline b(s) 为什么能降方差而不引入偏差？
<details><summary>答案</summary>

E[∇log π · b(s)] = b(s)·E[∇log π] = 0（概率归一化的推论），所以减去任何不依赖动作的 b 不改变期望，但让"权重" Q−b 围绕 0 波动，显著降低梯度估计的方差。实践中 b=V(s)，权重变成 advantage（Ch08）。
</details>

4. REINFORCE 为什么要等整条轨迹结束才更新？这带来什么缺点？
<details><summary>答案</summary>

回报 G_t 要到回合结束才知道。缺点：高方差（Monte Carlo）+ 必须完整回合 + 数据只能用一次（on-policy）。Ch08 的 critic 用 V 的自举估计替掉真实 G_t，把这三个问题都缓解。
</details>

### Ch08 Actor-Critic + GAE

1. Actor-Critic 里 actor 和 critic 各是什么、各自怎么更新？
<details><summary>答案</summary>

actor = 策略网络 π(a|s;θ)，沿 ∇log π·A 梯度上升；critic = 价值网络 V(s;φ)，用 TD/MC 回报做回归目标、最小化 MSE（它只是普通的监督回归）。两者共用数据互不求梯度。
</details>

2. 写出 GAE 定义 Â_t = Σ (γλ)^l δ_{t+l}，并说明 λ 的两个极端。
<details><summary>答案</summary>

δ_t = r + γV(s_{t+1}) − V(s_t) 是 TD error。λ=0 → Â=δ_t（一步，低方差高偏差）；λ=1 → Â = G_t − V(s_t)（MC advantage，无偏高方差）。λ 在 0.95 附近是常见甜点——和 Ch04 的 TD(λ) 数学结构完全相同。
</details>

3. 为什么用 advantage 而不是原始回报当策略梯度权重？
<details><summary>答案</summary>

advantage = 该动作比平均好多少：正的强化、负的抑制。比"绝对回报"方差小（回报整体都正时也能区分好坏动作），也比纯 TD error 更完整（GAE 综合多步信息）。
</details>

### Ch09 TRPO + PPO

1. TRPO 解的是什么约束优化问题？为什么需要信任域？
<details><summary>答案</summary>

max E[代理目标] s.t. KL(π_old, π_new) ≤ δ。策略一步走太远时，采样数据的分布和新策略失配，代理目标（局部近似）不再可信——限制每步 KL 就是在限制"近似还成立"的范围。
</details>

2. PPO-Clip 如何不用约束求解器达到类似效果？写出 clip 目标。
<details><summary>答案</summary>

L = E[ min( ratio·Â, clip(ratio, 1−ε, 1+ε)·Â ) ]，ratio = π_new/π_old。Â>0 时 ratio 上限 1+ε（再好也不无限增概率）、Â<0 时下限 1−ε——两支路都封住"一步走太远"，实现一阶优化的软信任域。
</details>

3. PPO 为什么能对同一批数据更新 K 个 epoch？哪个机制保证这不跑偏？
<details><summary>答案</summary>

importance ratio 修正新旧策略的概率差 + clip 限制单步偏离 + 通常还监控 approx KL（超阈值提前停 epoch）。这三个保险让"数据复用 K 次"的样本效率增益不破坏 on-policy 的正确性。
</details>

4. approx_kl 监控什么？它突然变大说明什么、该怎么办？
<details><summary>答案</summary>

E[(r−1) − log r]，k3 估计（总是 ≥0 且数值稳定）。变大 = 新策略离采样策略太远（epoch 数过多或 lr 过大）——标准做法是提前终止本轮 epoch（本课程 rlhf/grpo 的 `target_kl` 早停就是这个）。
</details>

---

## Phase 3

### Ch10 TinyGPT

1. 注意力的 softmax 为什么要除以 √d_k？
<details><summary>答案</summary>

q·k 的方差随 d_k 线性增长（Var(q·k)=d_k，单位方差分量时）；分数一大 softmax 就饱和成 one-hot，梯度消失。除 √d_k 把方差归一回 1。
</details>

2. causal mask 是怎么实现的？没有它会怎样？
<details><summary>答案</summary>

注意力分数矩阵的上三角（未来位置）填 −inf，softmax 后权重为 0，位置 i 只看 ≤i。没有它模型可以直接"偷看答案"，训练时 loss 虚低、生成时完全错误（生成只能自左向右）。
</details>

3. SFT 的 loss 为什么只在 response token 上算？
<details><summary>答案</summary>

我们要模型**学会生成回答**而不是复述 prompt：prompt 部分的预测误差和任务无关，mask 掉才不会稀释梯度（teacher forcing 下每个位置预测下一个 token，prompt 位置只作为条件输入）。
</details>

4. 用 RL 的语言重新描述语言模型：state / action / 转移分别是什么？
<details><summary>答案</summary>

state = 当前前缀 (x, y_<t)；action = 下一个 token；转移 = 把 token 拼进前缀（**确定性**！环境的随机性只在策略采样里）。这是 Ch12 把 PPO 搬到 LLM 上的全部桥梁。
</details>

### Ch11 Reward Modeling

1. Bradley-Terry 模型把"P(y_w ≻ y_l)"写成什么？怎么从三个公理推出来？
<details><summary>答案</summary>

P(y_w ≻ y_l) = σ(r(y_w) − r(y_l))。由单调性（分数高者胜）、对称性（平移不变）、Luce 选择公理（相对偏好只依赖分数差）可以证明偏好概率必须取 logistic 形式——损失只需要 reward 差，绝对刻度不可辨识（要加正则/归一化）。
</details>

2. RM 的 loss 只依赖 r(y_w)−r(y_l)。这个不可辨识性会带来什么问题？
<details><summary>答案</summary>

偏好数据只约束分数差，整体平移/放大不受约束——RM 可以把所有分数无限拉高同时保持排序不变。这就是"reward hacking/过优化"的空间：KL 惩罚（Ch12）正是把 policy 拉回 reference 来约束这个漂移。
</details>

3. 什么是 reward over-optimization（Goodhart）？Ch12 的哪个机制抑制它？
<details><summary>答案</summary>

policy 无限优化**代理** reward（RM 的打分）时，真奖励先升后降——RM 在训练分布外的外推会被钻空子。抑制机制：每个 token 加 β·KL(π‖π_ref) 惩罚，把 policy 锁在参考模型附近（β 越大锁得越紧）。
</details>

### Ch12 RLHF-PPO（InstructGPT 配方）

1. RLHF 三阶段是什么？为什么需要三个而不是端到端一个？
<details><summary>答案</summary>

SFT（学会基本语言/指令格式）→ RM（把人类偏好变成可微分数）→ PPO 微调（优化 RM 分数 − KL 惩罚）。偏好标注比"写出完美答案"容易得多；把"什么好"（RM）和"怎么生成"（policy）分开，人类只需要做相对判断。
</details>

2. token 级 reward 怎么构造？逐项说明。
<details><summary>答案</summary>

每个 response token：r_t = −β·(log π(a_t|s_t) − log π_ref(a_t|s_t))（KL 惩罚逐 token 分摊）；最后一个 token 位置再加 RM 对整个 response 的总分。序列级的 KL = 各 token KL 之和，序列级 RM 分数放到末尾——credit assignment 交给 GAE。
</details>

3. value head 怎么从 decoder 里读出 V(s_t)？为什么取位置 t 而不是最后一位？
<details><summary>答案</summary>

在最后一个 block 的隐状态上接一个 LayerNorm+Linear 头；causal attention 保证位置 t 的隐状态只含 (x, y_≤t)，正好是"预测 a_t 时的 state"。取最后一位就变成只对序列末尾有值。
</details>

### Ch13 GRPO（DeepSeek-R1 核心）

1. GRPO 去掉了 PPO-RLHF 的什么组件？用什么替代？
<details><summary>答案</summary>

去掉 critic/value 网络（省一半显存与一套优化）。替代：同一个 prompt 采 G 个 response，用组内奖励的 (r − mean)/std 作为 advantage——组均值就是 V(x) 的蒙特卡洛估计。
</details>

2. 为什么组标准化是 (r−r̄)/σ_r 而不是 r−r̄ 就够了？
<details><summary>答案</summary>

不同 prompt 的奖励尺度可能差很多（有的题全组都 0 分有的全组接近满分）。除以 σ_r 把各组的 advantage 方差归一，梯度尺度稳定；σ→0 时加 ε 保护（全同分组 advantage 置 0，符合"组内无信息"）。
</details>

3. G 越大 advantage 估计越准。代价是什么？（两个）
<details><summary>答案</summary>

① 每个 prompt 要采 G 条完整 response，rollout 计算 ×G；② 组内样本来自同一 prompt，估计的方差下降但不消除（prompt 间的泛化不靠 G）。DeepSeekMath 用的典型 G 是 8~64。
</details>

4. PPO 的 ratio/clip 机制在 GRPO 里还在吗？怎么在的？
<details><summary>答案</summary>

在：GRPO 仍是 on-policy PPO 更新——同样算 ratio = π_new/π_old、同样 clip(1±ε)、同样跑 K 个 epoch + approx KL 早停。它只换掉了 advantage 的来源（critic → group baseline），优化器部分原封不动。
</details>

### Ch14 DPO / KTO

1. DPO 的核心洞察是什么？它绕开了 RLHF 的哪一步？
<details><summary>答案</summary>

"你的语言模型 secretly 就是一个 reward model"：最优策略下的 reward 可以写成 r(x,y) = β log(π/π_ref) + β log Z(x)。把它代回 Bradley-Terry loss，偏好概率只依赖两个 log 概率比——直接对策略做偏好数据的极大似然，**不再需要显式 RM、不再需要 rollout/PPO**。
</details>

2. DPO 损失里的 log π(y_w|x)−log π_ref(y_w|x) 和 RLHF 的 KL 惩罚是什么关系？
<details><summary>答案</summary>

同一个东西：DPO 把"RL 最大化 r − β·KL"的最优解反代回 loss，隐式地完成了 KL 约束下的优化。β 同时控制偏离参考模型的强度——β 大离 π_ref 近、保守；β 小放得开、容易过拟合偏好噪声。
</details>

3. KTO 和 DPO 的数据需求差在哪？什么时候 KTO 更合适？
<details><summary>答案</summary>

DPO 需要成对偏好 (y_w, y_l)；KTO 只需要单条样本的好/坏标签（thumbs up/down），损失用前景理论的价值函数（对损失侧更敏感）+ 参考模型锚定的 KL 平衡项。线上只有点赞/点踩信号、凑不出配对时 KTO 更合适。
</details>

### Ch15 终局项目

1. 一次完整的"从语料到对齐模型"流水线要经过哪几步？每步用什么？
<details><summary>答案</summary>

预训练（TinyGPT next-token）→ SFT（指令-回答对，response-only loss）→ RM（偏好对，Bradley-Terry）→ GRPO（RM 分数 − βKL，组基线 advantage）。对应 InstructGPT/DeepSeek-R1 的完整配方。
</details>

2. 走完整个项目后，回看 Ch07 的策略梯度定理：它在 GRPO 里具体落在哪个量上？
<details><summary>答案</summary>

∇J = E[∇log π(y|x) · Â(x,y)]——GRPO 的实现就是它：∇log π 是每个 response token 的 log-prob 梯度之和，Â 是组标准化的 advantage。从老虎机到 LLM，这行公式没变过，变的是 π 变成了 transformer、 Â 的估计方式换成了组基线。
</details>

---

## Phase 4

### Ch16 PRM（过程奖励模型）

1. ORM 和 PRM 的打分粒度差在哪？各自的 Best-of-N 怎么用分数？
<details><summary>答案</summary>

ORM 只对**完整答案**打一个分（outcome）；PRM 对**每个推理步骤**打分（process）。BoN 时 ORM 选总分最高的候选；PRM 把各步分数聚合（求和/取 min/末步值），能区分"答案对但推理错"和"推理全对"——多步推理任务里 PRM 更可靠。
</details>

2. 为什么 PRM 的训练数据更难造？Ch16 用了什么技巧造负样本？
<details><summary>答案</summary>

人工标注每一步的好坏比标注最终答案贵得多。技巧：在正确的多步解上**人工制造错误的中间步骤**（make_wrong_step_variations），自动得到"同样的前缀、下一步对/错"的对比标签，省去人工逐步标注。
</details>

3. 举一个 ORM 会误判、PRM 不会的具体例子。
<details><summary>答案</summary>

两步加法题：候选 A 第一步就加错、最后答案碰巧对了（前面错误被后续错误抵消）；候选 B 每步都对只是最后抄写错。ORM 给 A 高分（只看结果）；PRM 看 A 的第一步就低分—— BoN 会选 B。
</details>

### Ch17 Self-Play + Constitutional AI

1. SPIN 的 GAN 式结构里 generator / discriminator 各是什么？"判别器=数据分布"意味着什么？
<details><summary>答案</summary>

generator = 当前的 policy（LLM）；discriminator = 判别"response 是真人数据还是模型生成"的分类器。SPIN 的洞察：当 policy 分布 = 数据分布 时判别器无法区分（最优判别器 = 1/2 处处），此时 policy 已学会目标分布、训练自然终止——不需要额外的 RM。
</details>

2. RLAIF 用什么替换了 RLHF 的人工标注？风险是什么？
<details><summary>答案</summary>

用 AI judge（通常由 constitution 规则约束的强模型）给 response 打偏好标签。风险：judge 的系统性偏差直接进入 reward——constitutional 原则的编写质量、judge 模型自身的能力上限都变成了对齐目标的上限，且错误会被 RL 放大。
</details>

3. Constitutional AI 的原则（constitution）在整个流程里出现在哪两处？
<details><summary>答案</summary>

① 监督阶段：让模型按原则 critique-and-revise 自己的回答，生成修订数据做 SFT；② RLAIF 阶段：judge 按原则逐条打分产生偏好对，再训 RM/做 RL。原则在两处都是"人类意图的可审计代理"。
</details>

### Ch18 Offline RL（CQL / IQL / DT）

1. Offline RL 和 online RL 的根本区别？为什么"直接在离线数据上跑 Q-learning"会失败？
<details><summary>答案</summary>

offline 数据集固定，不能再交互采样新数据。失败原因：Q-learning 的 max_a' Q(s',a') 会对**数据里没执行过的动作**给出任意外推值（OOD 动作），而离线设置永远无法通过尝试来纠正这个高估——bootstrapping 沿着虚假的高值不断自我强化，Q 爆炸。
</details>

2. CQL 和 IQL 各用什么机制抑制 OOD 外推？
<details><summary>答案</summary>

CQL：在标准 Bellman loss 上加正则——**压低数据外动作的 Q、抬高数据内动作的 Q**（保守到不冒进）。IQL：干脆不显式 max——用 expectile 回归（τ>0.5 的 asymmetric loss）只从**数据中见过的动作**里隐式逼近 max，天然不评估 OOD 动作。
</details>

3. Decision Transformer 完全不用 value function。它拿什么当条件？
<details><summary>答案</summary>

return-to-go R̂_t（从 t 到回合结束的实际累计奖励）：把 (R̂_t, s_t, a_t) 当序列喂给 transformer，因果地预测 a_t。推理时把 R̂ 设成目标值，模型"朝着这个回报走"——RL 问题被转写成条件序列建模，类似 prompt 工程。
</details>

4. expectile 回归的 τ 趋近 1 时，IQL 学到的东西趋向于什么？
<details><summary>答案</summary>

τ→1 时 asymmetric loss 只惩罚低估，估计量趋向条件**最大值** max_a Q(s,a)（数据分布内）——IQL 用一个连续的旋钮从"平均值"(τ=0.5) 平滑过渡到"近似 max"，绕开了显式 max 的 OOD 问题。
</details>

---

## Phase 5

### Ch19 Agent 基础（让 LLM 学会使用工具）

1. Agent loop 和 RL loop 的对应关系是什么？工具增强解码和普通自回归解码的差别在哪一步发生？
<details><summary>答案</summary>

观察（用户消息/工具返回）= 状态，动作 = 生成的 token（或一次工具调用），环境 = 搜索/计算器/沙盒，奖励 = 任务结局。多轮下 state 包含**工具注入的观察**——上下文随交互增长，且环境注入的 token 属于观察（state），**不算进 log π**（不是模型的动作）——这是 Agentic GRPO 最容易踩的坑。
</details>

2. Ch13 的 GRPOTrainer 为什么不能直接用于多轮 agent 训练？
<details><summary>答案</summary>

rollout 是"一口气采样 prompt→response"；多轮需要**交互循环**：模型段 ⇄ 环境段穿插（`decode_with_tools` 的结构）。log π 只累计模型 emit 的 token；advantage/clip/KL 部分原封不动。
</details>

3. RAFT 和 GRPO 的本质区别是什么？RAFT 什么条件下有效？
<details><summary>答案</summary>

RAFT = 把 group advantage 二值化（正确 1 / 错误 0）后做 SFT——丢掉"差多少"的信息和负样本梯度。有效条件：SFT 平均水平远低于 oracle best-of-K、且模型还有泛化余量（推理链类任务）；在模型已背熟训练集、贴着泛化天花板时会失效甚至倒退（Ch19 的玩具如实演示了这一点）。
</details>

4. 为什么「同一个模型 + 计算器工具」能赢「同一个模型裸算」？这对真实 LLM 意味着什么？
<details><summary>答案</summary>

网络只需学会抄写操作数、转写工具结果——序列模型的本行；而裸算要把进位加法压进参数——容量的硬功夫。工具是**能力边界的扩展**而非接口：o3/R1 的搜索-推理-回溯是同一原理的工业版。多数投票治随机错不治系统错、oracle best-of-K 是 RL 的理论收益上限——这两个仪表盘读数是 agent 工程的常识。
</details>

### Ch20 Agentic GRPO 实战（多轮强化学习 + 全书终章）

1. 多轮轨迹里 log π 怎么算？为什么环境注入的 token 不能计入？
<details><summary>答案</summary>

只对模型 emit 的 token 求和（rollout 时记 emitted 蒙版）。环境注入的 token 属于**观察**（state 的一部分）而非动作——把观察算进 π 的分子，策略梯度就错了（对不是自己做的选择求了梯度）。
</details>

2. 为什么 GRPO 在背熟的题集上没有梯度？
<details><summary>答案</summary>

组优势 Â=(r−r̄)/σ：背熟的题组内全对 → σ=0 → Â≡0 → 梯度为零。RL 的信号不在奖励本身，而在**组内的不一致**——「够得着但不稳」的任务才有学习信号（对应真实 RLHF 里按难度筛题、R1 的课程设计）。
</details>

3. 「无限题海」实验揭示了 on-policy RL 的什么本质？它和 SFT/RAFT 的数据观有何不同？
<details><summary>答案</summary>

SFT/RAFT 被固定数据集锁死；on-policy RL 的训练集可由任务分布**无限自生成**——模型始终在自己的能力边缘练习（自生成课程）。AlphaZero 自我对弈、R1 无限题海、Search-R1 真实搜索都是同一原理：把「训练数据」变成「训练环境」。
</details>

4. Agentic GRPO 相对 Ch13 单轮 GRPO 的三个结构变化是什么？
<details><summary>答案</summary>

① rollout 从一次采样变成模型段⇄环境段的交互循环；② 状态包含工具返回的观察（上下文增长）；③ log π 只对模型 emit 的 token 计算（观察不算）。目标函数（组优势/clip/KL）一字不改。
</details>

---

## 里程碑自测（Phase Gates）

### Phase 1 → 2 门槛

不看笔记回答：

1. 默写贝尔曼期望方程和 Q-learning 更新式
2. 用一句话向朋友解释 on-policy vs off-policy
3. MC、TD(0)、TD(λ) 的 bias/variance 关系

三条都过 → Ch05b（如需）→ Ch06。任何一条卡住 → 回对应章节。

### Phase 2 → 3 门槛

1. 默写策略梯度定理 + 说明 baseline 为什么无偏
2. GAE 的 λ 在调和什么矛盾？写出 δ_t
3. PPO-clip 在 Â>0 和 Â<0 时分别怎么"刹车"？

### Phase 3 → 4 门槛

1. RLHF 三阶段 + 每阶段的数据形态
2. GRPO 的 advantage 公式 + 它为什么不需要 critic
3. DPO 为什么可以完全跳过 rollout？

三条全过，恭喜——你已经能读懂 DeepSeek-R1 / InstructGPT 论文的方法部分了。

---

## 学不下去的时候

- **数学卡住**：跳过推导看结论 + 数值验证 cell，下一章会用更直观的方式再讲一遍（γ、baseline、GAE 都是螺旋式重复的）。
- **代码跑不通**：Restart Kernel and Run All Cells（90% 的 notebook 问题都是执行顺序问题）。
- **概念太抽象**：回到 `rlenvs/` 里把环境当玩具玩——打印 `env.P`、`env.R`，手动 step 几步，比再读一遍推导有用。
- **想知道"这在 LLM 里对应什么"**：每章小结都有 forward link； impatient 的话直接跳 Ch12 §12.4 的 MDP↔LLM 对照表。
