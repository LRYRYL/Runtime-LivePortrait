# 第三方许可与限制

本文件说明 `Runtime-LivePortrait` 依赖的第三方组件及其许可条件。
**如果你打算使用或分发本项目，请先读第 1 节 —— 它决定了整个项目不能商用。**

---

## 1. ⚠️ InsightFace 人脸模型：仅限非商业研究

本项目运行**必需**的人脸检测与关键点模型来自 **InsightFace `buffalo_l`**：

```
LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx
LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx
```

InsightFace 官方 README 原文（<https://github.com/deepinsight/insightface>）：

> The code of InsightFace is released under the MIT License. There is no limitation
> for both academic and commercial usage.
>
> **The training data containing the annotation (and the models trained with these
> data) are available for non-commercial research purposes only.**
>
> Both manual-downloading models from our github repo and **auto-downloading models
> with our python-library follow the above license policy**（即非商业研究）.

### 这意味着什么

| | 结论 |
|---|---|
| InsightFace **代码** | MIT，可商用 |
| InsightFace **模型** | ❌ **仅非商业研究** |
| 本项目整体 | ❌ **不能商用**（因为依赖上述模型）|

**两个常见误解，都不成立：**

1. ❌ *"我不打包模型，让用户自己下就没问题了"* —— 条款明确覆盖自动下载这条路，
   最终用户仍然受非商业限制。
2. ❌ *"我的代码是 MIT，所以整个项目是 MIT"* —— MIT 只覆盖**你写的代码**，
   不能给第三方模型授权。

**正确的做法是在 README 顶部显著声明非商业限制**（本仓库已这样做）。

> 如果你需要商用：InsightFace 官方给出了商业授权联系方式
> （`recognition-oss-pack@insightface.ai`，见其 README 的 License 一节），
> 或改用许可更宽松的人脸检测模型替换这两个 ONNX。

---

## 2. LivePortrait

- 仓库：<https://github.com/KwaiVGI/LivePortrait>
- 代码许可：**MIT**，© 2024 Kuaishou Visual Generation and Interaction Center
- 完整许可证文本见 `LivePortrait/LICENSE`

**代码是 MIT，可商用。** 但其官方权重包中**包含**第 1 节所述的 InsightFace 模型，
所以**打包好的官方权重整体受非商业限制**。

本项目**不随仓库分发权重**，改由 `tools/download_weights.py` 下载，
下载源为官方 Hugging Face 仓库 `KwaiVGI/LivePortrait`，逐个文件校验 SHA-256。

---

## 3. 本仓库自己写的代码

| 路径 | 许可 |
|---|---|
| `app.py` | MIT（见根目录 `LICENSE`）|
| `fastportrait/` | MIT |
| `tools/`（除 `download_weights.py` 下载的第三方文件）| MIT |
| `Start_WebUI.bat` | MIT |

---

## 4. 运行期依赖（Python 包）

均为宽松许可，可自由使用；此处列出以便审计：

| 包 | 许可 |
|---|---|
| PyTorch | BSD-3-Clause |
| torchvision | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| OpenCV (`opencv-python`) | Apache-2.0 |
| onnxruntime-gpu | MIT |
| pywin32 | PSF |
| tyro / rich / tqdm / PyYAML / scipy / scikit-image / albumentations / imageio | 各自宽松许可（MIT / BSD / Apache-2.0）|

> 注意 `onnxruntime-gpu` 被固定为 **1.18.0**：1.19+ 需要 CUDA 12 运行库。

---

## 5. 关于底图照片（重要）

**本仓库不附带任何人脸图片**，你必须自己提供底图。

这一点是刻意的，原因是"能生成一张脸"不等于"能分发这张脸"：

- 从数据集里取的真实人脸照片 → 涉及**照片中本人的肖像权**，
  这与数据集本身的许可（如 FFHQ 的 `CC BY-NC-SA 4.0`）是**两回事**
- 用 StyleGAN2-FFHQ 之类模型生成的合成脸 → 受**模型自身许可**约束
  （NVIDIA StyleGAN2-ADA 为 [NVIDIA Source Code License-NC](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html)，
  第 3.3 条限定"仅限非商业研究或评估"），且模型输出的法律地位本身有争议
- 二次元角色形象 → 可能涉及**原作的角色设计权利**，即使图片是 AI 生成的

**请只使用你拥有权利的照片**，并自行承担相应责任。

---

## 6. 伦理与法律

人脸动画与换脸技术受多国法规约束，包括但不限于：

- 欧盟《人工智能法案》(AI Act) 对深度伪造的透明度要求
- 中国《互联网信息服务深度合成管理规定》《生成式人工智能服务管理暂行办法》
- 美国各州关于未经同意制作他人肖像视频的法律
- 日本《刑法》关于名誉毁损与肖像权的相关规定

**使用前请确认你有权使用所涉及的人脸，并遵守当地法律。**
不得用于欺诈、冒充、骚扰、制作未经同意的私密内容等用途。

---

## 7. 免责声明

本项目按"现状"提供，不附带任何明示或暗示的担保。
作者不对因使用本项目而产生的任何直接或间接损失负责，
也不对使用者的行为及其法律后果负责。
