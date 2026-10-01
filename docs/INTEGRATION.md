# 接入 Deep-Live-Cam / 任意宿主

## 为什么必须走子进程

Deep-Live-Cam 的 venv 与本项目**无法共存于一个解释器**：

| | Deep-Live-Cam venv | lpv 环境 | 兼容？ |
|---|---|---|---|
| torch | **无** | 2.1.0+cu118 | ❌ LivePortrait 必需 |
| numpy | **2.5.3** | 1.26.4 | ❌ LivePortrait 依赖要求 1.x |
| opencv | 4.14.0 | 4.10.0 | ⚠️ |

把 torch 装进 Deep-Live-Cam 的 venv 会改动你已有的项目，违反"不干扰已有项目"的约束。
所以采用**与 HairFastGAN 相同的桥接模式**：
宿主用 `lpv` 解释器启动 worker，通过 stdin/stdout 传帧。

## 协议（全小端）

```
宿主 → worker   uint32 len | JPEG bytes        一帧 BGR
                uint32 0xFFFFFFFF              关闭
worker → 宿主   uint32 len | JPEG bytes        动画后的一帧
```

用 JPEG 而非原始像素：1280×720 原始是 2.7 MB，编码后约 60 KB。
编解码各约 1 ms，远低于动画本身的 ~68 ms。

**启动握手**：worker 会先加载模型（约 5–7 s），期间 LivePortrait 通过 `rich`
把加载日志打到 **stdout**。所以 READY 行**不是第一行**——
宿主必须**跳过非 READY 行**直到看到 `READY 1`（`frame_client.py` 已实现）。

## 用法

```python
import sys
from pathlib import Path

# 不写死路径：本项目就是当前仓库，tools/ 是它的子目录。
# 若宿主在别处，把 LPR 改成指向 Runtime-LivePortrait 即可。
LPR = Path(__file__).resolve().parent / "Runtime-LivePortrait"
sys.path.insert(0, str(LPR / "tools"))

from frame_client import FrameWorker

worker = FrameWorker(source=Path("portraits/sample.jpg"))   # 相对路径或调用方传入
if not worker.start():            # 阻塞到模型加载完（约 5-7 s）
    raise RuntimeError(worker.error)

while True:
    frame = get_camera_frame()    # BGR uint8
    out = worker.process(frame)   # 无人脸的帧会原样回传，不会失步
    if out is None:
        break                     # worker.error 里有原因
    show(out)

worker.stop()
```

## 在 Deep-Live-Cam 里挂成帧处理器

Deep-Live-Cam 的处理器接口（`modules/processors/frame/core.py` 的
`ALLOWED_PROCESSORS`，以及 `modules.globals.fp_ui`）需要一个模块，暴露
`NAME`、`pre_check`、`pre_start`、`process_frame`、`process_image`、`process_video`。

**没有替你写进 Deep-Live-Cam 的目录**，因为那会改动你要求不要动的项目。
如果要挂上去，在宿主项目的 `modules/processors/frame/` 下新建
`lpv.py`，大致如下（`process_frame_v2` 是 GUI 实时路径实际调用的那个）：

```python
import modules.globals
from tools.frame_client import FrameWorker    # 或把它复制进该项目

NAME = "DLC.LIVEPORTRAIT"
_WORKER = None

def pre_check(): return True

def pre_start():
    """模型加载 5-7 s，必须在点 Live 之前完成，不能在首帧里做。"""
    global _WORKER
    if _WORKER is not None: return True
    src = modules.globals.liveportrait_source_path
    if not src: return False
    _WORKER = FrameWorker(src)
    return _WORKER.start()

def process_frame(source_face, temp_frame): return temp_frame

def process_frame_v2(temp_frame):
    if _WORKER is None or not _WORKER.ready: return temp_frame
    out = _WORKER.process(temp_frame)
    return temp_frame if out is None else out

def process_image(image): return image
def process_video(*a, **k): pass
```

然后在 `ALLOWED_PROCESSORS` 里加上 `"lpv"`，并给
`modules.globals.fp_ui` 增加对应开关。

## 实测的桥接开销

| 路径 | 每帧 |
|---|---|
| 同进程（`tools/verify_speed.py`）| **67.6 ms → 14.75 FPS** |
| 子进程桥接（`tools/frame_client.py`）| **88.9 ms → 11.2 FPS** |

**桥接开销约 21 ms/帧**（JPEG 编解码 + 进程间传输）。
JPEG 质量从 90 降到 70 只能省约 4 ms，**说明开销主要在 IPC 本身而不是压缩**。
这 21 ms 在动画本身 68 ms 的衬托下是可接受的，但它是真实成本，不是零。

## 已知限制

1. **无人脸的帧会原样回传**，宿主不会失步。
2. **同步阻塞**：`process()` 会阻塞宿主线程约 89 ms。在 Deep-Live-Cam 的
   GUI 线程里直接调用会掉帧，建议放在采集线程（其 `_ProcessingWorker`
   本来就是独立线程）。
3. 源图更换需重启 worker（模型只加载一次）。
