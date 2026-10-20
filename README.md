# Video to Model

基于 Video Depth Anything 的视频深度生成项目。项目使用 Python 3.12 和 `uv` 管理环境，原始视频放在 `data/inputs/`，生成结果默认写入 `data/outputs/`。

项目会自动下载所选权重、初始化模型、处理视频并生成深度可视化视频。模型加载和分块推理均显示进度条。

## 环境准备

在项目根目录打开 PowerShell：

```powershell
uv python pin 3.12
uv sync
```

确认命令行入口：

```powershell
uv run video-to-model --help
```

## 运行平台

GPU 推理只在 **Linux x86_64** 与 **Windows x86_64** 上可用，macOS 只能跑 CPU：

| 平台 | 可用 `--device` | CUDA 12.4 构建 | xFormers |
|---|---|---|---|
| Linux x86_64 | `cuda` / `cpu` | ✅ 官方 cu124 wheel | ✅ |
| Windows x86_64 | `cuda` / `cpu` | ✅ 官方 cu124 wheel | ✅ |
| macOS（Apple Silicon） | 仅 `cpu` | ❌ 无 cu124 wheel | ❌ 官方不发布 macOS wheel |
| Linux aarch64（ARM） | 仅 `cpu` | ❌ 无 cu124 wheel | ❌ |

关于 xFormers（上游用来做 memory-efficient attention 的加速库）：

- 官方只发布 `manylinux_x86_64` / `win_amd64` wheel，**没有 macOS 版本**（PyPI 上只有 sdist，在 mac 上源码构建会失败）。因此 `pyproject.toml` 用 PEP 508 marker 把它限定在 Linux/Windows x86_64，macOS 上 `uv sync` 会直接跳过它，而不是报错。
- 在 macOS / ARM 上运行时，日志会出现 4 行 `xFormers not available`（来自 `dinov2_layers/attention.py`、`dinov2_layers/block.py`、`motion_module/attention.py`、`motion_module/motion_module.py` 的 import 探测）。这是提示不是错误，深度结果不变；代价是回退到普通注意力实现，失去 xFormers 省显存与提速的效果，加大 `--input-size` 时更容易碰显存上限。
- xFormers 只影响 GPU 上的注意力开销，`--device cpu` 本来就用不到它。

推荐部署环境是 Linux x86_64 GPU 服务器，见下一节。

## CUDA 版本：固定 12.4

项目固定使用 CUDA 12.4 的 PyTorch 官方构建，**不使用 CUDA 13**：cu13 构建要求较新的驱动，并且已经放弃 Maxwell/Pascal/Volta 架构（GTX 10 系、T4、V100 等小显卡服务器跑不了）；cu124 构建覆盖 sm_50 及以上，驱动要求低得多。

- 版本锁定：`torch==2.6.0` / `torchvision==0.21.0`。官方只为这两个版本发布 cu124 wheel，所以 torch 不能继续用更新的 2.x（2.13+ 只提供 cu126/cu130/cu132）。
- 取包来源：Linux x86_64 与 Windows x86_64 从 `https://download.pytorch.org/whl/cu124` 拉 `+cu124` wheel；`xformers==0.0.29.post3`（依赖严格对应 `torch==2.6.0`）也从同一索引安装。macOS、Linux aarch64 等无 cu124 wheel 的平台自动回退清华 PyPI 镜像。配置见 `pyproject.toml` 的 `[[tool.uv.index]]` 与 `[tool.uv.sources]`。
- CUDA 运行库：`nvidia-*-cu12==12.4.*`（cuBLAS 12.4.5.8、cudart 12.4.127、cuDNN 9.1.0.70、NCCL 2.21.5）+ `triton==3.2.0`，均由 `uv sync` 自动装好，不需要在服务器上单独安装 CUDA Toolkit。
- 驱动要求：`nvidia-smi` 显示的 CUDA Version ≥ 12.4（Linux 驱动 ≥ 550）。

在 GPU 服务器上安装并验证：

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

预期输出形如 `2.6.0+cu124 12.4 True`。若第三项是 `False`，先检查 `nvidia-smi`；若 `torch.__version__` 没有 `+cu124` 后缀，说明当前机器不是 Linux/Windows x86_64（macOS 开发机就是这种情况，用 `--device cpu` 即可）。

若服务器访问 `download.pytorch.org` 超时，可临时注释掉 `[tool.uv.sources]` 中的两条映射并重新 `uv lock`：清华镜像上 `torch==2.6.0` 的 Linux wheel 依赖同为 CUDA 12.4，结果等价。

## 输入视频

把原始视频放入：

```text
data/inputs/
```

当前示例视频：`data/inputs/MOV_03_17s.mp4`

## 快速运行

选择模型时建议使用明确的 `--model` 名称：

```powershell
uv run video-to-model data/inputs/MOV_03_17s.mp4 --model video-depth-anything-base
uv run video-to-model data/inputs/MOV_03_17s.mp4 --model metric-video-depth-anything-base
```

支持 CPU 和 CUDA（12.4，见上一节）推理。默认使用 CPU，因此没有 NVIDIA GPU/CUDA 环境时直接运行即可；有 CUDA 环境时可显式指定 `--device cuda`：

```powershell
uv run video-to-model data/inputs/MOV_03_17s.mp4 --model video-depth-anything-base --device cuda
```

六个可选模型名称：

```powershell
video-depth-anything-small
video-depth-anything-base
video-depth-anything-large
metric-video-depth-anything-small
metric-video-depth-anything-base
metric-video-depth-anything-large
```

程序第一次初始化指定模型时，会从对应 Hugging Face 仓库下载权重到 `models/` 下相应目录；后续运行复用已下载的权重。可以通过 `--checkpoint` 使用已有的本地权重。

### Hugging Face 下载超时

项目默认将 Hugging Face 元数据连接超时设为 60 秒、文件下载超时设为 600 秒，并自动重试 3 次。若当前网络无法访问 Hugging Face，可在运行前将 `HF_ENDPOINT` 设置为你所在网络可访问的 Hugging Face Hub 镜像/端点。例如 WSL bash：

```bash
export HF_ENDPOINT="https://<可访问的-Hugging-Face-Hub-端点>"
uv run video-to-model data/inputs/MOV_03_17s.mp4 --model video-depth-anything-base --device cuda
```

本次报错是连接 Hugging Face 时超时，权重文件尚未开始传输。修复网络/端点后重跑同一命令即可；下载器会复用已完成的本地文件。

## 长视频与内存（分批推理）

上游实现会先把整段视频解码驻留内存，再把所有窗口的深度图累积在一个 list 里，最后一次性归一化渲染 —— 峰值内存随帧数线性增长（约 `帧数 × 宽 × 高 × 15` 字节，1.2 万帧 @1280×720 就需要 ~150 GB），长视频会被系统 OOM 直接杀掉（进程无任何报错，`echo $?` 为 137）。

本项目把推理改成了常数内存的流式管线，结果与上游逐位一致（`scripts/verify_streaming.py` 对拍验证）：

- 视频按需逐帧解码，不再全量常驻；
- 每个窗口的深度图算完立即写入内存映射文件（memmap），堆内存只保留滑窗缓冲与对齐所需的常数帧；
- 可视化拆成两阶段：先分块扫描求全局明暗范围，再分块编码 mp4。

因此**内存不再随视频长度增长**，代价是需要磁盘：深度原始值默认保留在 `<输出名>.mmap`（fp16 约 每千帧 @1280×720 占用 1.8 GB），配合同名 `.json` 元数据。

```bash
# 原始深度值只跑一次，之后反复换参数重渲染（不再占用 GPU）
uv run video-to-model --render-depth-map data/outputs/MOV_03_17s_depth.mmap --output data/outputs/vis_gray.mp4

# 不需要保留原始深度时，让跑完自动删除
uv run video-to-model data/inputs/MOV_03_17s.mp4 --model video-depth-anything-base --device cuda --discard-depth-map
```

默认 fp16 会损失数值精度；如果单帧深度值超出 fp16 表示范围（约 65504，常见于几乎静止的片段使对齐拟合病态），该帧会存为 `inf` 并不参与明暗统计，程序会输出溢出帧数。需要完整数值时用 `--depth-dtype fp32`（磁盘翻倍）。

如果只是想降低资源占用而不关心原始深度，直接抽帧更便宜：`--target-fps 10 --max-res 640` 可把帧数和每帧字节数同时降几倍，内存与耗时一并下降。

两种实现的逐项内存构成、峰值公式、实测数字以及后续可以继续压降的手段，见 [docs/memory-optimization.md](docs/memory-optimization.md)。

## 常用参数

| 参数 | 用途 | 示例 |
|---|---|---|
| `input` | 输入视频路径 | `data/inputs/MOV_03_17s.mp4` |
| `--output` | 指定输出路径 | `--output data/outputs/MOV_03_17s_depth.mp4` |
| `--model` | 明确指定模型类型与规格 | `--model video-depth-anything-base` |
| `--device` | 推理设备：`cpu` 或 `cuda`，默认 `cpu` | `--device cuda` |
| `--checkpoint` | 指定已有权重文件 | `--checkpoint models/path/to/model.pth` |
| `--input-size` | 模型输入尺寸，正数且为 14 的倍数 | `--input-size 518` |
| `--fp32` | 关闭 FP16 autocast（只影响推理精度，不影响磁盘深度格式） | `--fp32` |
| `--streaming` | 实验性逐帧流式模式：用 temporal cache 代替滑窗对齐，显存最省、速度最快，但精度下降（上游 d1 从 0.926 降到 0.836），且上游实现没有 metric 分支 | `--streaming` |
| `--max-len` | 最多解码多少帧后停下，`-1` 不限（默认 `-1`） | `--max-len 3000` |
| `--target-fps` | 抽帧目标帧率，`-1` 保留原帧率（默认 `-1`） | `--target-fps 10` |
| `--max-res` | 解码时长边最大像素，同时决定输出深度分辨率（默认 `1280`） | `--max-res 640` |
| `--depth-map` | 原始深度 memmap 的存放路径 | `--depth-map data/outputs/run.mmap` |
| `--depth-dtype` | 深度落盘精度：`fp16`（默认，省一半空间）或 `fp32` | `--depth-dtype fp32` |
| `--discard-depth-map` | 渲染完成后删除原始深度文件（默认保留） | `--discard-depth-map` |
| `--render-depth-map` | 只重新渲染已有的深度文件，跳过推理（此时可省略 `input`） | `--render-depth-map data/outputs/x.mmap` |
| `--video-crf` | 可视化 mp4 的 H.264 质量档，**越大体积越小**（默认 `18` 接近无损；`23` 约省一半，`28` 约省三分之二） | `--video-crf 23` |
| `--audio` | 布尔值控制是否把源视频音轨写进输出视频：`--audio`、`--audio true` 开启，`--audio false` 或省略则关闭（画面流 `-c:v copy` 不重编码，音频转 AAC） | `--audio true` |
| `--colormap` | 可视化配色；`gray` 输出传统**灰度**深度视频，`gray_r` 明暗反向（默认 `inferno`，即热成像风格） | `--colormap gray` |

## 运行进度

模型权重下载显示下载进度，模型初始化显示阶段进度，推理阶段按帧显示进度，渲染阶段额外显示两个进度条：`统计明暗范围`（第一趟扫描深度文件）与 `编码视频`（第二趟读回并写 mp4）。总数在容器不报告帧数时为上界估计。

可视化 mp4 默认使用 CRF 18（接近视觉无损），体积通常比手机压出来的源视频大很多；要缩小用 `--video-crf 23`（约省一半）或 `28`（约省三分之二）。已有的深度文件可直接降低质量重渲染，不必重跑推理：

```bash
uv run video-to-model --render-depth-map data/outputs/视频片段_depth.mmap --video-crf 23 --output data/outputs/视频片段_depth_小.mp4
```

## 命令模板 / 占位符

将尖括号中的内容替换成自己的值：

```powershell
uv run video-to-model <输入视频路径> --model <模型名称> [--device <cpu|cuda>] [--output <输出视频路径>] [--checkpoint <权重文件路径>] [--input-size <尺寸>] [--max-len <帧数>] [--target-fps <帧率>] [--max-res <分辨率>] [--depth-dtype <fp16|fp32>] [--discard-depth-map] [--fp32] [--streaming]
```

例如：

```powershell
uv run video-to-model data/inputs/<视频文件名>.mp4 --model <模型名称> --output data/outputs/<输出文件名>_depth.mp4
```

PowerShell 中不要原样输入尖括号占位内容；请替换成真实路径或选项值。省略 `--output` 时，默认目标为 `data/outputs/<输入文件名>_depth.mp4`。

## 输出视频的音轨

深度视频的画面是逐帧重新绘制的新视频流，所以**默认没有声音**（不是丢了，而是本来就没有）。“画”和“声”是两条独立的流，需要显式合并：

```bash
uv run video-to-model data/inputs/视频片段.mp4 --model video-depth-anything-base --device cuda --audio
```

加 `--audio` 后，编码完成会把源视频的第一条音轨混进输出文件：画面流直接 `-c:v copy` 不重编码（**不造成任何画质损失、几乎不耗时**），音频转成 AAC 以兼容 mp4 容器，并用 `-shortest` 对齐时长。若源视频本身无声轨，会打印 `源视频没有音轨，输出保持静音` 并正常完成。

它是布尔开关。`--audio` 与 `--audio true` 等价（表示需要声音）；`--audio false` 或省略该参数则表示不需要：

```bash
uv run video-to-model 输入.mp4 --model video-depth-anything-base --device cuda --audio        # 有声
uv run video-to-model 输入.mp4 --model video-depth-anything-base --device cuda --audio true   # 有声
uv run video-to-model 输入.mp4 --model video-depth-anything-base --device cuda --audio false  # 无声（默认）
```

已有的深度文件也可以补上音轨（不需要重跑推理）：`--render-depth-map` 会从深度文件的元数据里找回录下的源视频路径，所以要求那个源文件仍在原位：

```bash
uv run video-to-model --render-depth-map data/outputs/视频片段_depth.mmap --audio --output data/outputs/视频片段_depth_有声.mp4
```

## 可视化配色（灰度深度视频）

输出视频看着像**热成像**是正常的：上游 `save_video` 写死了 matplotlib 的 `inferno` 色表，那本来就是给热图/密度图用的感知均匀色表，不是本项目选的。现在可以用 `--colormap` 换掉：

```bash
# 传统灰度深度视频（近/远用明暗表示）
uv run video-to-model --render-depth-map data/outputs/视频片段_depth.mmap --colormap gray --output data/outputs/灰度深度.mp4
```

可选值：`inferno`（默认，热图风）、`gray`（灰度）、`gray_r`（灰度反向）、`viridis`、`magma`、`plasma`、`turbo`、`hot`。

两点要记住：

- 映射规则始终是 **“深度值越大越亮”**，与色表无关；换色表**不改变任何深度数值**（`.mmap` 内容完全一致）。
- “亮是近还是远”取决于模型类型：相对深度模型（`video-depth-*`）数值大通常表示更近；metric 模型输出的是距离值，数值大表示更远。拿不准就取一帧看已知远近物体的数值大小即可。
- 灰度视频只是可视化产物。要拿数值做 3D，请直接读 `.mmap`，不要用任何 mp4 反解。

## 模型仓库

- 相对深度：Small、Base、Large，分别对应 `vits`、`vitb`、`vitl`。
- 度量深度：使用相同规格并附加 `--metric`。
- 许可可能因仓库和规格而异，请在使用前查看所选 [Hugging Face 模型仓库](https://huggingface.co/depth-anything)。

## 项目结构

```text
models/       模型文件与自动下载的权重
src/          Python 包、CLI、模型注册和推理适配边界
scripts/      开发验证脚本（与上游逐帧对拍）
data/inputs/  原始视频
data/outputs/ 深度生成结果（.mp4 可视化 + .mmap 原始深度值）
docs/         目录架构规划与资源/性能对照
```

目录规划详见 [docs/directory-structure.md](docs/directory-structure.md)。
