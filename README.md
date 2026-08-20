# 强化学习系统学习教材

> 一套**从零基础到能用 PPO / GRPO 训练 LLM** 的交互式 Jupyter Notebook 教材。
> 强调数学完整推导 + 交互式可视化 + 自研可观测内部状态的小环境。

## 适用人群

- 已会 Python，**对强化学习基本零基础**
- 希望理解贝尔曼方程、策略梯度、PPO 等核心算法的**数学原理**
- 最终目标是能**读懂和改写 PPO / GRPO 在 LLM（RLHF、DeepSeek-R1 风格）上的训练代码**

## 学习路径

```
Phase 1：经典 RL 基础 (Ch00-05)         <-- 全部已交付
    ↓
Phase 2：策略梯度 + PPO (Ch06-09)        <-- 全部已交付
    ↓
Phase 3：LLM RLHF + GRPO (Ch10-15)     <-- 全部已交付（终极目标）
    ↓
Phase 4：研究前沿 (Ch16-18)             <-- 全部已交付
```

> **当前进度**：Phase 1-4 共 **19 章 / 157 个测试**已全部交付。仓库内任何一条路径都能直接跑通。

### 章节地图

> 标 **已交付** = 该章 notebook + 配套 `build_chXX.py` + 测试 + 可视化承诺全部到位。

| 章节 | 主题 | 实际兑现的关键可视化 |
|---|---|---|
| **Ch00** | 环境搭建 + RL 全景 | 点击式 ClickWorld、学习路径图 **（已交付）** |
| **Ch01** | 多臂老虎机 | ε 滑块 + 200-seed 基准 + UCB1 后悔界推导 **（已交付）** |
| **Ch02** | MDP + 贝尔曼方程 | γ 滑块 → V 热力图 + 值传播动画 **（已交付）** |
| **Ch03** | 动态规划 | DP 扫描动画 + 策略箭头 **（已交付）** |
| **Ch04** | TD 学习 | MC vs TD 收敛曲线 + (α, λ) 双滑块 **（已交付）** |
| **Ch05** | Q-learning / SARSA | CliffWalk 30-seed 对比 + maximization bias 实验 **（已交付）** |
| **Ch05b** | PyTorch 速成（过渡章） | sin 拟合训练曲线 + 有限差分梯度验证 **（已交付）** |
| **Ch06** | DQN + 函数逼近 | Q 值热力图 + 训练曲线 + replay buffer 动画 + Double/Dueling 对比 **（已交付）** |
| **Ch07** | 策略梯度定理 | 策略分布 + baseline 方差对比 + CartPoleLite 策略热力图 **（已交付）** |
| **Ch08** | Actor-Critic + GAE | 三联动画（state / V / π）+ GAE λ 滑块 + advantage 分解 **（已交付）** |
| **Ch09** | TRPO + PPO | PPO-Clip 经典 4 子图 + KL 散度监控 + 训练曲线对比 **（已交付）** |
| **Ch10** | TinyGPT 从零搭 | 多头注意力热力图 + 训练 loss 曲线 + 采样对比 **（已交付）** |
| **Ch11** | Reward Modeling | 偏好 UI + Bradley-Terry 拟合 + reward 过优化曲线 **（已交付）** |
| **Ch12** | RLHF-PPO (InstructGPT) | 4 模型仪表盘 + token 级 reward 分解 + β 扫描 **（已交付）** |
| **Ch13** | **GRPO (DeepSeek-R1)** | group advantage 柱状图 + GRPO vs PPO 对比 + 训练曲线 **（已交付）** |
| **Ch14** | DPO / KTO | 算法决策树 + DPO/KTO/PPO 边界对比 + reward-margin 散点 **（已交付）** |
| **Ch15** | 终局项目 | SFT warmup 曲线 + GRPO dashboard + 完整学习路径图 **（已交付）** |
| **Ch16** | PRM (Process Reward Model) | step-level reward 热力图 + PRM vs ORM Best-of-N 对比 **（已交付）** |
| **Ch17** | Self-Play + Constitutional AI | SPIN GAN 结构 + AI judge 评分分布 + RLAIF-GRPO vs Human-GRPO 对比 **（已交付）** |
| **Ch18** | Offline RL (CQL/IQL/DT) | OOD Q 值发散对比 + expectile 数值演示 + DT return-conditioning 扫描 **（已交付）** |

> **项目完成度：100%（20 个 notebook / 157 测试，全部 Phase 1-4 已交付）**

> 📖 **配套学习指南**：`STUDY_GUIDE.md` —— 全部章节的自测题（答案可折叠）、
> 三个 Phase 门槛自测、每章时间预估。**建议每学完一章就做对应自测，过关再前进。**

### Fast-track 路径（~20 小时直达 GRPO）

如果你已有一定基础、想尽快到达 LLM RLHF 部分：

**Ch00 → Ch01 → Ch05 → (Ch05b) → Ch07 → Ch09 → Ch13**

> **现已全部可走通**：Phase 1-3 全部交付后，这条 fast-track 上的每一章都可直接打开 `notebooks/chXX_*.ipynb` 学习，配套测试全绿。没用过 PyTorch 的读者把 **Ch05b** 插在 Ch07 前面（约 1 小时）。

## 安装与使用

> **学生请直接看 [GETTING_STARTED.md](GETTING_STARTED.md)**（三步走 + 常见问题表）；
> 老师查看 `doctor.py`（环境体检，产出可回传的诊断报告）与 `setup_windows.bat` / `setup-mac-linux.sh`（一键配置）。

### 1. 安装依赖

```bash
cd RLStudy
pip install -r requirements.txt
```

> **交互式绘图说明**：动画用 `to_jshtml()` 生成 HTML 播放器、滑块用 ipywidgets 控件，
> 普通 inline 后端即可，**不需要** `%matplotlib widget`。ipywidgets 缺失时
> `make_interactive` 会自动降级为静态图（打印提示）。

### 2. 验证环境

```bash
pytest tests/
```

如果 **157 个测试全绿**（共 14 个测试文件，覆盖全部 env / 模型 / 算法），说明所有 env + TinyGPT + RM + RLHF + GRPO + DPO/KTO + PRM + Self-Play + Offline RL 都能正常导入和使用。

### 3. 打开第一课

```bash
jupyter lab notebooks/
```

从 `ch00_setup_and_overview.ipynb` 开始按顺序学习；每学完一章，做 `STUDY_GUIDE.md` 里对应的自测题。

## 设计原则

1. **数学完整推导**：用三层呈现 —— 主线只展示结论、可折叠 `<details>` 给完整证明、配套数值验证 cell
2. **可视化四件套**：matplotlib 动画/曲线 + ipywidgets 滑块 + 自研迷你环境 + LLM 训练模拟
3. **自研环境暴露内部状态**：`P[s,a,s']` 转移张量、`Q` 表、奖励函数等都可访问，这是和 Spinning Up / HF 课的最大差异
4. **numpy → PyTorch 渐进式切换**：Phase 1（Ch00-05）纯 numpy 实现表格方法，把认知负载留给 RL 本身、不被 autograd 干扰；**Ch05b（PyTorch 速成）** 承接过渡；Phase 2（Ch06-09）从 **DQN 起切入 PyTorch**——Q 网络的梯度回传交给 `loss.backward()`，但训练 loop、replay buffer、PPO clip 仍手写；Phase 3（Ch10-15）全面 PyTorch 化，TinyGPT / RewardModel / RLHF-PPO / GRPO 都用 `nn.Module` 搭建，但仍**不依赖黑盒 RL 库**（如 stable-baselines3 / TRL），保证每一步算法都可读、可改。

## 教材特色

- ✅ **每章独立可跑**：单章 notebook 在普通笔记本上 < 10 分钟跑完
- ✅ **每个算法从零实现**：不依赖黑盒库，但用 PyTorch 求导
- ✅ **关键公式有数值验证**：用有限差分对比解析梯度等手段
- ✅ **交互组件有静态降级方案**：ipywidgets 缺失时自动降级为静态图（已实现到 `make_interactive` 内部）
- ✅ **每章配练习 + 自测题**：Ch01-05 练习配 `solutions/` 参考答案；Ch06-13 各章末尾有 📝 练习（可折叠提示 + 预期结果）；全部章节的自测题在 `STUDY_GUIDE.md`（答案可折叠，支持检索式学习）

## 参考资料

**教材 & 综述**
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- OpenAI Spinning Up
- Lilian Weng 博客：*A (Long) Peek into Reinforcement Learning*、*Policy Gradient Algorithms*

**Phase 1（Ch00-05）经典 RL**
- Sutton, *Learning to predict by the method of temporal differences* (1988) —— TD(λ)，Ch04 主参考
- Watkins, *Learning from delayed rewards* (1989) —— Q-learning 原始论文，Ch05

**Phase 2（Ch06-09）深度 RL + 策略梯度**
- Mnih et al., *Human-level control through deep reinforcement learning* (Nature 2015) —— **DQN**，Ch06 主参考
- Hasselt et al., *Deep Reinforcement Learning with Double Q-learning* (2016) —— **Double DQN**，Ch06 §6.5
- Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning* (2016) —— **Dueling DQN**，Ch06 §6.5
- Williams, *REINFORCE* (1992) —— 策略梯度定理源头，Ch07
- Sutton et al., *Policy Gradient Methods for Reinforcement Learning with Function Approximation* (2000) —— Ch07
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015) —— **GAE**，Ch08 主参考
- Schulman et al., *Trust Region Policy Optimization* (TRPO, 2015)
- Schulman et al., *Proximal Policy Optimization Algorithms* (**PPO**, 2017) —— Ch09 主参考
- Schulman, *Approximating KL Divergence* (blog, 2016) —— Ch09 §9.x

**Phase 3（Ch10-15）LLM + RLHF + GRPO**
- Vaswani et al., *Attention Is All You Need* (2017) —— Transformer / self-attention，Ch10 灵魂等式
- Ouyang et al., *Training language models to follow instructions with human feedback* (**InstructGPT**, 2022) —— Ch12 主参考（RLHF 三阶段：SFT + RM + PPO）
- Christiano et al., *Deep Reinforcement Learning from Human Preferences* (2017) —— RLHF 起源，Ch11
- Bradley & Terry, *Rank Analysis of Incomplete Block Designs* (1952) —— Bradley-Terry 偏好模型，Ch11
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning* (**GRPO**, 2024) —— Ch13 主参考
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability via RL* (2025) —— Ch13 / Ch15 主参考
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (**DPO**, 2023) —— Ch14
- Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization* (**KTO**, 2024) —— Ch14

## 目录结构

```
RLStudy/
├── README.md                       # 本文件
├── STUDY_GUIDE.md                  # 学习指南：全章自测题（答案可折叠）+ Phase 门槛 + 时间预估
├── WRITING_STYLE.md                # 写作风格指南：教材的叙事与交互设计规范（改内容前必读）
├── requirements.txt
├── build_notebooks.py              # 总入口：重建注册的 15 个 notebook（ch00-14）+ 5 个 solutions
│                                   #   python build_notebooks.py             # 全部
│                                   #   python build_notebooks.py --chapters 06-09  # 只 Phase 2
│                                   #   python build_notebooks.py --chapter 13     # 只 GRPO
│                                   #   python build_notebooks.py --list
│                                   #   ch15 手维护；ch16-18 用各自的 build_chXX.py 重建
├── build_ch06.py ... build_ch18.py # 各章 notebook 内容定义（独立可跑、被 build_notebooks.py 调用；
│                                   #   ch17/ch18 的内容在 ch17_content.txt / ch18_content.txt）
├── nb_helpers.py                   # 全部 build 脚本共享的构建工具（统一 cell helper / 写出 / metadata）
├── conftest.py + pytest.ini        # pytest 配置
├── rlenvs/                         # 自研迷你环境（纯 Python 包，可 from rlenvs import GridWorld）
│                                   #   GridWorld / CliffWalk / RandomWalk / MultiArmedBandit
│                                   #   CartPoleLite (Phase 2) / TinyGPT (Phase 3)
├── utils/                          # 共享可视化与工具（set_seed / torch_utils / ...）
├── notebooks/                      # 20 个 ipynb（ch00 ~ ch18 + ch05b PyTorch 速成）
├── solutions/                      # 5 个练习参考答案 ipynb（ch01-thompson / ch02-bridge-γ / ...）
├── tests/                          # 14 个测试文件，共 157 个用例
│                                   #   envs: bandit / grid_world / cliff_walk / random_walk / cart_pole_lite / environment
│                                   #   models: tiny_gpt / reward_model / rlhf / grpo / dpo
│                                   #   frontier: prm / self_play / offline_rl
├── data/                           # 训练数据
│   └── tiny_corpus.txt             #   Ch10 TinyGPT 的合成训练语料
└── assets/                         # 生成的图、动画、checkpoint
    ├── ch12_*.png                  #   Ch12 RLHF-PPO 仪表盘 + β 扫描（4 张）
    ├── ch13_*.png                  #   Ch13 GRPO group advantage / dashboard / vs PPO（3 张）
    ├── ch15_*.png                  #   Ch15 capstone SFT warmup / GRPO dashboard / 学习路径图（4 张）
    ├── figures/                    #   其它静态图（matplotlib savefig 输出，按需生成）
    └── checkpoints/                #   模型权重（训练产物，按需生成）
```

## License

仅供个人学习使用。
