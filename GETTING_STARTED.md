# 学生上手指南（GETTING_STARTED）

> 写给第一次配置环境的同学。全部搞完约 10~20 分钟（主要在等下载）。
> 任何时候卡住：跑一遍 `python doctor.py`，把生成的 `doctor_report.txt` 发给老师。

## 三步走

### 第 1 步：拿到项目

```bash
git clone <老师发的仓库地址>
cd RLStudy
```

（不会 git？在仓库网页点绿色 **Code → Download ZIP**，解压后进入文件夹也一样。）

### 第 2 步：一键配置环境

**Windows**：双击 `setup_windows.bat`，等它跑完（会下载约 300MB 依赖，别关窗口）。

**macOS / Linux**：终端里执行：

```bash
bash setup-mac-linux.sh
```

脚本会自动：找 Python → 创建独立虚拟环境 `.venv`（**不污染**你系统里的 Python）→ 安装 CPU 版 PyTorch 和全部依赖 → 跑一遍环境体检。

### 第 3 步：开始学习

配置成功后 JupyterLab 会自动打开（没弹出就手动访问 http://localhost:8888）。

以后每次学习**不需要**重跑 setup 脚本，只要：

- **Windows**：双击 `start_windows.bat`（或命令行里 `.venv\Scripts\activate` 然后 `jupyter lab notebooks/`）
- **macOS / Linux**：`source .venv/bin/activate && jupyter lab notebooks/`

从 `ch00_setup_and_overview.ipynb` 开始，**每章先 Restart Kernel and Run All Cells**（菜单 Kernel 里）。每章 <10 分钟能跑完，做完对应练习，再去 `STUDY_GUIDE.md` 做自测题。

## 常见问题

| 症状 | 原因与解法 |
|---|---|
| 提示「没找到 Python」 | 没 装 Python 或没加 PATH：到 [python.org](https://www.python.org/downloads/) 装 3.10+，**第一屏勾选 "Add Python to PATH"**，重跑脚本 |
| Python 版本低于 3.10 | 装 3.10+；若系统里有多个版本，Windows 用 `py -3.12 -m venv .venv` 手动建环境 |
| PyTorch 下载特别慢 | torch 走的是官方源确实较慢，挂上校园网/热点耐心等；实在不行先跳过（前五章纯 numpy 不需要它），后面单独执行 `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| 其它包下载慢 | 可用国内镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 公司/校园代理拦下载 | `pip install --proxy http://代理地址:端口 ...`，或换手机热点 |
| 体检有 ✗ | 按 ✗ 旁边的「修复建议」处理；不行就发 `doctor_report.txt` 给老师 |
| notebook 里图不显示 / NameError | 九成是 cell 没按顺序跑：**Restart Kernel and Run All Cells** 一次 |
| 电脑实在带不动 / 不想装环境 | 找老师开云端方案（GitHub Codespaces：仓库页绿色 Code 按钮 → Codespaces） |

## 环境体检（随时可跑）

```bash
python doctor.py
```

检查 Python 版本、9 个依赖、PyTorch、Jupyter、项目包、真实冒烟测试（跑几步环境/一次网络前向/一次画图）和性能抽测——报告写在 `doctor_report.txt`，有问题发这个文件就行，不用截图描述。
