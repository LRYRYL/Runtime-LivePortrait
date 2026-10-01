# 暂停时的状态（第 16 轮末）

## 一句话

**功能全部可用；速度目标未达成。暂停在一个"有明确下一步"的位置。**

---

## 已完成且经验证

| 项 | 状态 | 证据 |
|---|---|---|
| 新文件夹 `Runtime-LivePortrait/`（不干扰其他项目）| ✅ | 今天被改文件全在此目录内 |
| 隔离环境 `lpv` | ✅ | torch 2.1.0+cu118 (cuda True)、numpy 1.26.4、opencv 4.10、ort 1.19.2 |
| 权重 628 MB（hf-mirror + curl，未耗 VPN）| ✅ | **8/8 校验通过** |
| 实时驱动 `fastportrait/realtime.py` | ✅ | 同进程 **~68 ms（14.6 FPS）** |
| 会话层 `fastportrait/session.py` | ✅ | 跳帧/统计逻辑已用静态帧验证 |
| 子进程桥接（Deep-Live-Cam 必需）| ✅ | 端到端验证，开销 **~15 ms** |
| 全链路验证工具 `tools/run_video.py` | ✅ | 两条路径均 PASS（无需真人）|
| 5D grid_sample 替代采样器 | ✅ | 数值等价 **max\|diff\| ≤ 4.8e-07** |
| ONNX 导出 | ✅ 导出成功过 | 实测比 PyTorch 慢，**模型文件已清理**（可随时用 `tools/export_onnx.py` 重建）|

## 未达成

**目标第 5 项"接近 Deep-Live-Cam 的 40–52 FPS"** —— 当前 ~14.6 FPS（约 37%）。

**且已用数据排除两条路**：
- `torch.compile`：Windows 不可用（torch 2.1 dynamo 限制）
- ONNX + CUDA EP：实测 **117.6 ms vs PyTorch 105.7 ms，更慢**

**唯一剩下的路是 TensorRT EP。**

---

## 暂停点：TensorRT 已推到"差一次实测"

本轮刚做的事：

| 步骤 | 结果 |
|---|---|
| 发现 ORT 1.18 面向 TRT 8.6，而本机 TRT 是 **10.16** → EP 无法注册 | ✅ 已定位 |
| 确认 ORT **1.19+** 支持 TRT 10 | ✅ |
| 升级 `onnxruntime-gpu` 1.18.0 → **1.19.2** | ✅ 成功，numpy 仍是 1.26.4 |
| 环境完整性复查（torch/cuda/opencv）| ✅ 未破坏 |
| 主路径回归 | ✅ RESULT: PASS |

### 下一步只有一件事：**实测 TensorRT EP 是否注册并提速**

```powershell
cd <Runtime-LivePortrait 目录>
$env:PATH = "$env:USERPROFILE\.conda\envs\personalive\Lib\site-packages\tensorrt_libs;$env:PATH"
# 然后把 TRT 的 dll 目录加进 PATH，跑：
& "$env:USERPROFILE\.conda\envs\lpv\python.exe" -c @"
import onnxruntime as ort
so = ort.SessionOptions()
s = ort.InferenceSession('onnx/warp_decode.onnx', so,
      providers=['TensorrtExecutionProvider','CUDAExecutionProvider'])
print(s.get_providers())
"@
```

把 `tensorrt_libs`（内含 `nvinfer_10.dll`）加进 PATH 是**关键** ——
之前 TRT EP 注册失败的原因就是找不到它。

**测试要点**：
1. TRT EP 是否**真的**被选中（`get_providers()` 第一位是 `TensorrtExecutionProvider`）
2. 首次运行会有引擎构建时间（可能数分钟），必须**先 warmup 再计时**
3. 用 `trt_fp16_enable=True`，并开 `trt_engine_cache_enable` 缓存引擎
4. 与 PyTorch 的 105.7 ms / 生产路径的 68 ms 对比

**若 TRT 也提不了速，则速度目标应判定为在该硬件上不可达。**

---

## 仍然悬空、只有你能回答的问题

**观感是否可用** —— 我问过多次未获回复，而它决定这条线是否值得继续。

- 图片：运行 `tools/render_angles.py` 或网页 UI 自行查看
- 实时：`tools/run_webcam.py --display --seconds 30`

**另外**：性能数字在 68 ms（第 4–6 轮）与 88–138 ms（第 8–16 轮）之间波动，
代码未变。已排除我的代码与残留进程；GPU 上只有系统组件（explorer、SearchHost、
msedgewebview2、NVIDIA Overlay）。**要拿可比基线，请在重启后重测。**

---

## 未做（需你授权）

**把处理器挂进宿主项目的 `modules/processors/frame/`** ——
会改动你要求不要动的项目，所以只写了方案与代码，见 `docs/INTEGRATION.md`。

---

## 文档索引

| 文件 | 内容 |
|---|---|
| `docs/PERFORMANCE.md` | 全部实测数字（含退化记录）|
| `docs/INTEGRATION.md` | 接入 Deep-Live-Cam 的方案与处理器代码 |
| `docs/ONNX_RESULT.md` | ONNX 导出成功但**不提速**的结论 |
| `docs/ONNX_BLOCKER.md` | 5D grid_sample 的完整探查史（含 4 次错误结论）|
