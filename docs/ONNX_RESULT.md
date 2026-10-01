# ONNX 导出：成功导出，但**没有提速**（第 15 轮）

## 结论摘要

| 项目 | 结果 |
|---|---|
| **替代采样器**（`grid_sample_3d`）| ✅ **数值等价**：真实形状 max\|diff\| **4.8e-07** |
| **ONNX 导出** | ✅ **成功导出过**（531.7 + 107.4 MB），但**实测更慢，文件已清理** |
| 导出后数值一致性 | ⚠️ max\|diff\| **3.6e-03**（相对 3.7e-03，超过 1e-3 阈值）|
| **速度** | ❌ **ONNX 比 PyTorch 更慢**：**117.6 ms vs 105.7 ms** |

**所以：阻塞了 11 轮的 5D grid_sample 问题解决了，但提速目标没有达成。**

---

## 一、采样器：终于做对了

关键突破是**放弃三轴链式 gather，改用扁平化单次 gather**：

```
input (B,C,D,H,W) -> (B,C,D*H*W)
线性索引  z*(H*W) + y*W + x    形状 (B,1,Dout*Hout*Wout)
一次 gather 沿最后一维          -> (B,C,Dout*Hout*Wout)
reshape                        -> (B,C,Dout,Hout,Wout)
```

**一个 gather、一维索引、没有轴顺序可算错。** 验证结果：

| 形状 | max\|diff\| |
|---|---|
| `(1,2,4,5,6)` | 2.384e-07 |
| `(1,4,16,8,8)` | 2.384e-07 |
| **`(1,32,16,64,64)`（真实）** | **4.768e-07** |

全部是 float32 噪声级。**这个"绕开问题"的建议是对的** —— 前 3 次失败都是因为我试图把 `(B,Dout,Hout,Wout)` 变换成三轴 gather 各自需要的轴顺序。

## 二、导出时踩到的两个坑（都已记录）

**坑 1：实例属性赋值不被 trace**
`warping.deform_input = lambda ...` 在 eager 下有效，但 `torch.onnx.export` trace 的是**类级 forward**，看不到实例属性，仍然报
`GridSample with 5D volumetric input`。

**坑 2：在 forward 里改 `__class__` 也太晚**
tracer 那时已经解析完调用。

**有效做法**：直接替换 `torch.nn.functional.grid_sample` 本身 ——
`WarpingNetwork.deform_input` 是按这个名字调用的，导出器在 trace 时解析该符号。
替换范围限定在导出调用内并总是还原。

## 三、为什么没有提速（关键负面结论）

ONNX Runtime 用的是 **CUDAExecutionProvider**（本机 TRT 是 v10，与 ORT 1.18 面向的 8.6 不匹配，无法用 TensorRT EP）。

**117.6 vs 105.7 ms** —— ORT 的 CUDA EP 在这个模型上**没有优势**。原因很可能是：
- PyTorch 的 cuDNN 卷积已高度优化
- ORT 每次 `sess.run()` 需要 CPU↔GPU 拷贝（numpy 输入），而 PyTorch 全程在 GPU 上

**这说明"导出到 ONNX 就能提速"是个错误假设** —— 真正的加速来自 **TensorRT**，而 TensorRT 这条路被版本不匹配堵住（升级 ORT 或 TRT 都有连带风险）。

## 四、另外注意：数值一致性 3.6e-03 超过阈值

`instance_norm is set to train=True` 的警告意味着导出以 train 模式处理了 instance norm，
这可能改变输出。**3.6e-03 虽小，但严格来说导出模型与 PyTorch 不等价**，
在采用前必须先把这一点查清（改 eval 模式或确认 SPADE 的 instance norm 行为）。

## 五、诚实的总账

**第 5 轮开始追 ONNX，到第 15 轮才导出成功，而结果是：不提速。**

中间 11 轮里，我：
- 错误断言过"z 轴不可解释"（后用 D≥4 标定推翻）
- 用错误的探针方法得出过"真三线性"的错误结论
- 在轴顺序上连续失败 3 次
- 一度把"维度数相同"误当成充分条件

**唯一正确的决策是最后那次"绕开而不是继续推导"。**

**这个结果应当如实告知：ONNX 这条路走通了，但它不解决问题。**
真正需要的是 TensorRT，而那条路被 TRT v10 与 ORT 1.18 的版本鸿沟挡住。
