# 部署说明

面向**没有配置过环境的 Windows 电脑**。目标是：`git clone` 之后**双击一次**就能跑起来。

---

## 一、需要什么

| 项目 | 要求 | 说明 |
|---|---|---|
| 系统 | **Windows 10/11 64 位** | 代码里有 Windows 专用的控制台处理 |
| 显卡 | **NVIDIA，显存 ≥ 6 GB** | A 卡 / 核显不行，依赖 CUDA |
| 驱动 | 支持 CUDA 11.8（≥ 452.39）| **不需要装 CUDA Toolkit** |
| 磁盘 | **约 9 GB** | 项目 47 MB + 依赖 3 GB + `.venv` 5 GB + 权重 667 MB |
| Python | **不需要预装** | 没有的话安装脚本会装一份（per-user）|
| 管理员权限 | **不需要** | |

> **关于 CUDA Toolkit**：不用装。torch 的 wheel **自带 cu118 运行库**
> （`cudart64_110.dll`、`cublasLt64_11.dll`、`cudnn64_8.dll`），
> `fastportrait/__init__.py` 启动时会自动把 `torch/lib` 加进 PATH。

---

## 二、一键安装

```bat
git clone <仓库地址>
cd Runtime-LivePortrait
一键安装.bat
```

`一键安装.bat` 调用 `tools\setup.ps1`，依次做六件事：

| 步骤 | 做什么 | 联网 |
|---|---|---|
| 1 | 找 Python 3.10；找不到就用**华为/清华镜像**下载安装器并静默装（`InstallAllUsers=0`，per-user）| 可能 |
| 2 | 在项目里创建虚拟环境 `.venv` | 否 |
| 3 | 下载依赖 wheel 到 `.venv-wheels`（约 3 GB）| 是 |
| 4 | 从本地 wheel 安装依赖（`--no-index --find-links`）| **否** |
| 5 | 下载模型权重到 `LivePortrait\pretrained_weights`（约 667 MB）| 是 |
| 6 | 校验：打印 torch / CUDA / numpy / opencv / onnxruntime 状态 | 否 |

**只有第 1、3、5 步需要网络，而且只发生一次。** 之后再跑安装会直接复用。

### 装完启动

```bat
Start_WebUI.bat
```

浏览器会打开 **http://127.0.0.1:7860**。想换端口：`Start_WebUI.bat 7861`。

---

## 三、分步做（不想用一键脚本时）

```powershell
# 1) 装 Python 3.10（若系统已有可跳过）
#    官方: https://www.python.org/downloads/release/python-31011/
#    国内: https://mirrors.huaweicloud.com/python/3.10.11/python-3.10.11-amd64.exe

# 2) 建虚拟环境
python -m venv .venv

# 3) 看你的显卡该用哪套 CUDA（cu118 还是 cu128）
python tools\gpu_choice.py

# 4) 下载依赖 wheel（几 GB，一次）—— 自动按显卡选，写进 .venv-wheels\<栈名>\
.venv\Scripts\python.exe tools\fetch_wheels.py

# 5) 从本地 wheel 安装（把 cu118 换成上一步选中的栈）
.venv\Scripts\python.exe -m pip install --no-index --find-links .venv-wheels\cu118 -r tools\requirements-cu118.txt
.venv\Scripts\python.exe -m pip install --no-index --find-links .venv-wheels\cu118 -r tools\requirements-common.txt

# 6) 下载权重
.venv\Scripts\python.exe tools\download_weights.py --mirror

# 7) 自检
.venv\Scripts\python.exe tools\check_env.py
```

> **顺序不能反**：先装 CUDA 栈，再装公共依赖。
> 因为 torch 会自己拉 `numpy 2.x`，而 LivePortrait **不兼容 numpy 2.x** ——
> `requirements-common.txt` 把 numpy 钉在 `1.26.4`，后装才能把它拉回来。
> `setup.ps1` 里就是这个顺序。

---

## 四、为什么有两套 CUDA（重要）

**没有任何一个 PyTorch 构建能同时把各种 N 卡都跑好。**

| 栈 | torch | CUDA | onnxruntime | 覆盖架构 |
|---|---|---|---|---|
| **cu118** | 2.1.0+cu118 | 11.8 | 1.18.0 | sm_50 ~ **sm_90**（原生 sm_86 / sm_89）|
| **cu128** | 2.7.0+cu128 | 12.8 | 1.22.0 | sm_50 ~ **sm_120**（原生 sm_86 / sm_120）|

**选错会怎样：**

- **RTX 50 系（sm_120）用 cu118** → **完全跑不了**：
  ```
  UserWarning: NVIDIA GeForce RTX 5070 ... with CUDA capability sm_120 is not
  compatible with the current PyTorch installation.
  The current PyTorch install supports CUDA capabilities sm_37 ... sm_90
  ```
  因为 **CUDA 11.8 在物理上无法编译 sm_120**。
- **RTX 40 系（sm_89）用 cu128** → 能跑，但 **cu128 没有 sm_89 的 cubin**，
  会退到 sm_86 二进制。**实测同一台 RTX 4080 Laptop：cu118 = 82.5 ms/帧，
  cu128 = 92.4 ms/帧，慢约 12%。**

**所以按显卡自动选：**

| 显卡 | 栈 |
|---|---|
| RTX 20 系 (sm_75) / A100 (sm_80) / **RTX 30 系 (sm_86)** / **RTX 40 系 (sm_89)** | **cu118** |
| H100 (sm_90) / B100 (sm_100) / **RTX 50 系 (sm_120)** | **cu128** |

规则在 `tools/gpu_choice.py`（安装脚本和下载脚本共用同一个判断，只有一份逻辑）。

**另外一个坑**：onnxruntime 的版本必须跟着 CUDA 栈走。`onnxruntime-gpu 1.18` 是
按 CUDA 11.8 + cuDNN 8 编译的，遇到 cu128 的 CUDA 12.8 + cuDNN 9 会报
`CUDA_PATH is set but CUDA wasnt able to be loaded`。所以 cu128 那套配 **1.22.0**。

---

## 五、脚本开关

`tools\setup.ps1` 支持：

| 参数 / 环境变量 | 作用 |
|---|---|
| `-Cuda cu118\|cu128\|auto` | 强制 CUDA 栈（默认 auto，按显卡检测）|
| `$env:LPR_CUDA` | 同上（`-Cuda` 优先）|
| `-SkipWeights` | 不下载权重（已有就跳过）|
| `-SkipWheels` | 只用现有 `.venv-wheels`，不下载 |
| `-Reinstall` | 删掉 `.venv` 重建 |
| `$env:LPR_PYTHON` | 指定用哪个 `python.exe` |
| `$env:LPR_SOURCE` | `cn`（默认，清华镜像）或 `pypi` |
| `$env:LPR_OFFLINE=1` | 完全离线，需要下载时直接失败 |

`tools\fetch_wheels.py` 支持：

| 参数 | 作用 |
|---|---|
| `--cuda auto\|cu118\|cu128` | 强制 CUDA 栈（默认 auto，按显卡检测）|
| `--source cn\|pypi` | 用哪个包索引 |
| `--check` | 只报告已下载情况，不下载 |
| `--dest <路径>` | wheel 放别处 |

**只下载当前显卡需要的那一套** —— 不会把 cu118 和 cu128 都拉下来。

---

## 六、仓库里为什么没有权重和 wheel

这是**刻意**的，不是漏了：

| 内容 | 为什么不在仓库里 |
|---|---|
| **模型权重**（667 MB）| 体积大；更重要的是 **InsightFace `buffalo_l` 仅限非商业研究**，**不是我们能再分发的**。让用户从上游 Hugging Face 自己取 |
| **依赖 wheel**（几 GB）| 第三方二进制，`pip` 从官方源取更合规、仓库也更小 |
| **Python 安装器** | 让用户从 python.org（或镜像）拿原件，避免转手分发 |

`download_weights.py` 会**逐个校验 SHA-256**，`fetch_wheels.py` 直接调用 `pip`，
两者都从**上游官方地址**取 —— 转手环节为零。

**仓库里只放清单（几百字节），不放二进制**：

```
tools/requirements-common.txt   与显卡无关的依赖
tools/requirements-cu118.txt    CUDA 11.8 栈（sm_75 ~ sm_89）
tools/requirements-cu128.txt    CUDA 12.8 栈（sm_90 及以上，RTX 50 必需）
tools/gpu_choice.py             按 compute capability 选栈（唯一判断来源）
```

许可细节见 [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md)。

---

## 七、故障排查

### 安装阶段

| 现象 | 原因 / 解决 |
|---|---|
| 双击闪一下就没了 | 右键用 PowerShell 跑 `tools\setup.ps1` 看报错；或先跑 `环境自检.bat` |
| `no Python 3.10 found` 但机器上有 Python | 版本不是 3.10。**必须 3.10**（3.11+ 有依赖不兼容）。用 `$env:LPR_PYTHON` 指定 |
| 安装 Python 卡很久 | 静默安装约 1–2 分钟。超过 5 分钟就手动装 3.10 再重跑 |
| 依赖下载失败 | 换源：`$env:LPR_SOURCE='pypi'`，然后重跑 |
| `venv creation failed` | 多半是 Microsoft Store 的 python 假冒版。装 python.org 的完整版 |
| 磁盘不够 | 需要约 10 GB。`fetch_wheels.py --dest D:\wheels` 可以把 wheel 放别的盘 |
| 下错了 CUDA 栈 | 见下方「CUDA 栈不对」 |

### CUDA 栈不对（N 卡型号相关）

| 现象 | 原因 / 解决 |
|---|---|
| `CUDA capability sm_120 is not compatible with the current PyTorch installation` | **RTX 50 系装了 cu118**。cu118 最高只到 sm_90，**物理上无法支持 sm_120**。重装成 cu128：见下方 |
| `cuda=False` 但显卡是 N 卡 | 同上，栈选错了。跑 `环境自检.bat`，它会直接告诉你该用哪个栈 |
| `CUDA_PATH is set but CUDA wasnt able to be loaded` | onnxruntime 与 CUDA 栈不匹配。cu118 配 1.18.0，cu128 配 1.22.0 —— 用清单文件装就不会错 |

**重装成正确的栈：**

```powershell
$env:LPR_CUDA='cu128'          # 换成 cu118 或 cu128
powershell -ExecutionPolicy Bypass -File tools\setup.ps1 -Reinstall
```

`-Reinstall` 会重建 `.venv`，`fetch_wheels.py` 会按新栈下载并放到
`.venv-wheels\cu128\`（两套互不干扰）。

想先确认该用哪套，不装任何东西：

```powershell
python tools\gpu_choice.py
```

### 运行阶段

| 现象 | 原因 / 解决 |
|---|---|
| `model weights are missing` | 权重没下全。`.venv\Scripts\python.exe tools\download_weights.py --mirror` |
| `LoadLibrary failed with error 126` | torch 与 CUDA 栈不匹配，见上 |
| 报缺 `ninja` / 编译扩展 | **与运行无关**，可忽略 |
| 网页能开但没画面 | 底图不是正脸，或摄像头前没人。换**正脸、中性表情**的照片 |
| 摄像头打不开 | 被微信/钉钉/浏览器占用，关掉它们 |
| 帧率只有约 11–12 | **正常**。瓶颈是模型（约 80–92 ms/帧），不是配置问题 |
| 帧率明显更低（如 5） | 先跑 `tools\speed_probe.py`（或双击 `测速.bat`）确认是显卡本身还是别的问题 |
| 关窗口后 `portraits\uploaded.jpg` 还在 | `pywin32` 没装上。`.venv\Scripts\python.exe -m pip install pywin32` |
| 端口被占用 | `Start_WebUI.bat 7861` |

### 重装依赖

```powershell
$stack = 'cu118'   # 或 cu128
.venv\Scripts\python.exe -m pip install --no-index --find-links ".venv-wheels\$stack" -r "tools\requirements-$stack.txt"
.venv\Scripts\python.exe -m pip install --no-index --find-links ".venv-wheels\$stack" -r tools\requirements-common.txt
```

---

## 七、关键版本（不要随意改）

| 包 | 版本 | 为什么 |
|---|---|---|
| Python | **3.10** | 依赖只在这个版本验证过 |
| torch / torchvision | **2.1.0+cu118 / 0.16.0+cu118** | 与 cu118 运行库匹配 |
| numpy | **1.26.4** | 2.x 与 LivePortrait 不兼容 |
| onnxruntime-gpu | **1.18.0** | **1.19+ 需要 CUDA 12**，会报缺 `cublasLt64_12.dll` |
| pywin32 | ≥ 306 | 关窗口时执行隐私清理 |

---

## 八、卸载

删掉项目文件夹即可。安装脚本**没有往系统里装东西**，
除了第 1 步那份 per-user 的 Python 3.10（在
`%LOCALAPPDATA%\Programs\Python\Python310`），
系统里本来就有 3.10 的话连这个都不会装。
