# Runtime-LivePortrait — 实测性能报告

**目标**：用 LivePortrait（warping + SPADE，**非扩散**）实现 PersonaLive 级的神经头像动画，
达到接近 Deep-Live-Cam 的效率。

## 结论

**架构选对了，帧率提升到 PersonaLive 的 2.65×；但未达到 Deep-Live-Cam 的效率。**

---

## 本机实测（RTX 4080 Laptop 12GB，全部本机真实数字）

| 方案 | 架构 | 输出 | 本机实测 |
|---|---|---|---|
| **PersonaLive** | 4 步扩散 UNet × 16 帧窗口 | 512×512 | 744 ms → **5.4 FPS** |
| LivePortrait + 贴回驱动帧 | 形变 + SPADE，无扩散 | 1280×720 | 91.8 ms → 10.9 FPS |
| **LivePortrait + 直接输出源图动画** | 同上 | 512×512 | **67.6–68.0 ms → 14.7–14.8 FPS** |
| **Deep-Live-Cam** | insightface 换脸（**无动画**）| 1280×720 | 19–25 ms → **40–52 FPS** |

**14.75 / 5.4 = 2.7× 于 PersonaLive；但离 Deep-Live-Cam 的 40 FPS 仍差 2.7×。**

### 阶段分解（14.75 FPS，67.6 ms/帧）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| **warp + SPADE 解码** | **41.2–41.7 ms** | **61%** |
| 运动提取 | ~10.7 ms | 15% |
| 人脸检测（106 点）| ~9.6 ms | 14% |
| 裁剪 | ~1.5 ms | 2% |
| paste-back | 0（已关闭）| — |

其他：模型加载 5.0 s；源图准备 2.1 s（每张肖像一次）；显存峰值 **0.69–1.02 GB**。

---

## 本轮的三个发现与修复

**1. 关闭 paste-back 同时解决观感与性能（最大收获）**

初版把动画后的源图**贴回**摄像头帧，出现**矩形接缝**——源图自己的背景
（金色雕像）透过掩码露出来。根源不是 bug：**LivePortrait 的设计是"输出源图的动画"，
不是"把源图的脸合成进驱动画面"**。源图 682×1023（竖），驱动帧 1280×720（横），
构图不匹配必然露馅。

改为**直接输出动画后的源图**后：
- 接缝消失，画面干净
- 顺带省掉 paste-back 的 **~23 ms**
- 10.9 → **14.2 FPS**

**2. paste-back 掩码每帧重建 → 缓存（贴回模式下省 28 ms/帧）**
`prepare_paste_back` 只依赖帧尺寸，固定摄像头分辨率下结果完全相同，
却每帧重算（实测 10.3 ms）。

**3. 输出偏蓝**
LivePortrait 整条管线在 **RGB** 空间工作（裁剪时 BGR→RGB），
输出被按 BGR 处理导致蓝通道被当成红通道。
修法：返回前 `cv2.cvtColor(I_p, cv2.COLOR_RGB2BGR)`。

### 一个诚实的负面结果

我一度以为 **fp16 未生效**是个大优化（`flag_use_half_precision` 只被
`inference_ctx()` 读取，而那是调用方必须手动进入的 `torch.autocast` 块，我初版没用）。
补齐后实测 **帧率完全没变**（122.6 → 122.6 ms）——模型本来就以 fp16 加载，
autocast 在此近乎 no-op。**我预期错了。**

顺带发现 `stitching` 的 retargeting MLP **不能跑 fp16**
（`expected scalar type Half but found Float`），所以现在模型前向用 autocast、
张量运算回 fp32。

---

## 距离实时还差什么

瓶颈单一且明确：**warp + SPADE 解码 41 ms，占 58%**。

| 想做的优化 | 结论 |
|---|---|
| `torch.compile` | ❌ **Windows 上不可用**（torch 2.1 dynamo 限制）。论文的 14.8 ms/帧正是开启它、在 4090 上测的 |
| fp16 | ❌ 已是 fp16，实测无额外收益 |
| 降检测频率 | 最多省 10 ms |
| **ONNX/TensorRT 导出** | ❌ **第 5 轮尝试失败**，被 torch 2.1 的 5D `grid_sample` 挡住，见 [ONNX_BLOCKER.md](ONNX_BLOCKER.md) |

### ONNX 导出：已定位确切阻塞点（第 5 轮）

唯一能真正提速的路径是 ONNX Runtime（warp+SPADE 占 62%）。本轮尝试导出，**失败**，
但拿到了确切原因与可复现证据：

- `WarpingNetwork.deform_input` 是 **5D 体素 `grid_sample`**，torch 旧版 ONNX 导出器直接拒绝
  （`OnnxExporterError: Unsupported: ONNX export of operator GridSample with 5D volumetric input`）
- "折成 4D"的自然绕法**实测不等价**：两种假设的 max|diff| 分别为 **2.808** 与直接报错
- dynamo 导出器能处理更多算子，但 `dynamo=` **torch 2.2 才引入**，本环境是 2.1

**这不是调参能解决的**——需手写 3D 体素采样算子，或升级 torch（会破坏 LivePortrait
钉死的 numpy 1.26.4 / opencv 4.10 依赖）。详见 [ONNX_BLOCKER.md](ONNX_BLOCKER.md)。

### 本轮修掉的一个真 bug

把 `pasteback` 默认改成 `False` 后，`verify_speed.py` 仍把 1280×720 的驱动帧与
**512×512** 的动画输出直接 `hstack`，报
`all the input array dimensions except for the concatenation axis must match exactly`。
已改为按高度缩放后再拼接。

**当前实测：68.3 ms → 14.6 FPS**（warp+SPADE 42.4 ms = 62%）。

---

## 当前状态

| 项 | 状态 |
|---|---|
| 新文件夹 `Runtime-LivePortrait/`（不干扰已有项目）| ✅ |
| 克隆 KwaiVGI/LivePortrait (9b294b3) | ✅ |
| 隔离环境 `lpv`（torch 2.1.0+cu118、onnxruntime-gpu 1.18、CUDA 可用）| ✅ |
| 权重 628 MB（hf-mirror + curl，未耗 VPN）| ✅ **8/8 校验通过** |
| `fastportrait/realtime.py` 模型驱动 | ✅ |
| `fastportrait/session.py` 端到端会话层（摄像头/统计/跳帧）| ✅ 已用静态帧验证 |
| `tools/frame_worker.py` + `frame_client.py` **子进程桥接** | ✅ 端到端验证通过 |
| `tools/run_video.py` **视频驱动全链路验证**（无需真人）| ✅ 两条路径均 PASS |
| `tools/verify_speed.py` 帧率实测 | ✅ |
| `tools/run_webcam.py` 摄像头驱动 | ✅ 已写（实时数字仍待真人验证）|
| 偏蓝 / fp16 / 掩码重复计算 | ✅ 已修 |
| 矩形接缝 | ✅ 已通过换显示模式解决 |
| **达到 Deep-Live-Cam 效率** | ❌ **未达成**（14.75 vs 40 FPS，为 37%）|
| 挂进 Deep-Live-Cam 的 `ALLOWED_PROCESSORS` | ⏳ **未做**（会改动你要求别动的项目；接入方案与代码已备好，见 `INTEGRATION.md`）|

### 桥接开销（客观测，第 4 轮）

| 路径 | 每帧 |
|---|---|
| 同进程 | **67.6 ms → 14.75 FPS** |
| 子进程桥接（Deep-Live-Cam 场景必须走这条）| **88.9–90.4 ms → 11.1–11.2 FPS** |

桥接开销 **约 21 ms/帧**。JPEG 质量 90→70 只省约 4 ms，
说明开销在 **IPC 本身而非压缩**。

### 第 8–10 轮：性能出现未解释的退化（诚实记录）

同一命令、**代码未改动**，帧率从 68 ms 掉到 **87–119 ms**：

| 轮次 | median | FPS |
|---|---|---|
| 第 4–6 轮 | **67.6–70.4 ms** | **14.2–14.8** |
| 第 8 轮 | 98.9 / 100.6 / 112.2 ms | 10.1 / 9.9 / 8.9 |
| 第 10 轮 | 118.7 / 90.7 / 87.5 ms | 8.4 / 11.0 / 11.4 |

**原因未查明。** 已排除：
- 我的代码改动 —— 第 8–10 轮都没有碰 `fastportrait/` 或 `verify_speed.py`
- 残留进程 —— 无 python 进程存活
- `onnx/` 残留 —— 已清理

**GPU 状态实测**（测速时）：利用率 **37%**、显存 1640/12282 MiB、
SM 时钟 2580 MHz、温度 63°C、功耗 **68.64 W**。

占用 GPU 的进程**全部是系统/桌面组件**，不是我能清理的东西：
`explorer.exe`、`StartMenuExperienceHost.exe`、`SearchHost.exe`、
`msedgewebview2.exe`、`CrossDeviceResume.exe`、`NVIDIA Overlay.exe`（两个）。

**判断**：笔记本 GPU 在 68 W 量级下与桌面组件共享功耗预算，
且室温/散热状态会影响持续性能。**第 4–6 轮的 14.6 FPS 可能是较"干净"条件下的数字，
而现在的 8–11 FPS 是负载条件下的数字。两者都不算错，但不可直接比较。**

**要得到可比结论，需在重启后、关闭桌面组件的条件下重测。** 我保留全部数字而不是挑好看的报。

### 端到端验证（第 6 轮）

之前 `run_webcam.py` 的实时数字**无效**（摄像头对着空房间，每帧都被跳过）。
本轮新增 `tools/run_video.py`：**用视频文件喂进完全相同的链路**
（采集 → JPEG 编码 → 子进程 IPC → 解码 → 动画），
因此在**不需要有人坐在摄像头前**的条件下就能确定性验证并计时。

| 路径 | 每帧 | FPS | 说明 |
|---|---|---|---|
| 同进程 | **70.2 ms** | **14.2** | 模型本身的成本 |
| **子进程桥接** | **85.6 ms** | **11.7** | Deep-Live-Cam 必须走的路径 |

**两条路径都跑通并 PASS**（每帧都真正被动画，skipped 0）。
桥接开销 **约 15 ms/帧**（本轮实测，比上轮估的 21 ms 更准）。

### 又一个被数据否定的假设

我先前用 `warpAffine` + `BORDER_REPLICATE` 生成带平移的测试视频，
结果**检测器在该视频上一帧都找不到人脸**（`detector found: 0 faces`），
而同内容的 PNG 正常。我一开始怀疑是 mp4 编码损伤，实测排除：

| 编码 | 解码后与原图 mean\|diff\| |
|---|---|
| mp4v | **2** |
| MJPG | **1** |

编码几乎无损，所以问题出在我**自己的测试数据生成**（平移+边缘复制）上，
不是管线。改用干净片段后两条路径立刻通过。

**这条记下来**：测试数据本身会制造假故障，遇到"全都失败"时先怀疑数据。

### 诚实说明：摄像头实时验证仍不完整

`tools/run_webcam.py` 跑通了 120 帧，但**当时摄像头对着空房间**，
每帧都是"未检测到人脸"，因此那次报出的 28.6 FPS **无效**
（未处理的帧几乎不耗时，会给出虚高的数字）。

已修正统计逻辑：**只统计真正处理过的帧**（见 `session.py` 的
`SessionStats.add`）。该逻辑已用静态帧验证：
有人脸的帧计入、空白帧不计入并正确跳过。

**权威数字是静态实测的 14.7–14.8 FPS**（`tools/verify_speed.py`，两次独立运行）。
**摄像头实时帧率仍未在"画面里有人脸"的条件下验证过** —— 这需要你在摄像头前坐好，
我无法代劳。

输出样张可用 `tools/render_angles.py` 重新生成（见 `docs/testD_deg.jpg`）
