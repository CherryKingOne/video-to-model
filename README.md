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

支持 CPU 和 CUDA 推理。默认使用 CPU，因此没有 NVIDIA GPU/CUDA 环境时直接运行即可；有 CUDA 环境时可显式指定 `--device cuda`：

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

## 常用参数

| 参数 | 用途 | 示例 |
|---|---|---|
| `input` | 输入视频路径 | `data/inputs/MOV_03_17s.mp4` |
| `--output` | 指定输出路径 | `--output data/outputs/MOV_03_17s_depth.mp4` |
| `--model` | 明确指定模型类型与规格 | `--model video-depth-anything-base` |
| `--device` | 推理设备：`cpu` 或 `cuda`，默认 `cpu` | `--device cuda` |
| `--checkpoint` | 指定已有权重文件 | `--checkpoint models/path/to/model.pth` |
| `--input-size` | 模型输入尺寸，正数且为 14 的倍数 | `--input-size 518` |
| `--fp32` | 关闭 FP16 autocast | `--fp32` |
| `--streaming` | 使用实验性流式模式 | `--streaming` |

## 运行进度

模型权重下载显示下载进度，模型初始化显示阶段进度，上游滑窗推理显示视频帧处理进度。

## 命令模板 / 占位符

将尖括号中的内容替换成自己的值：

```powershell
uv run video-to-model <输入视频路径> --model <模型名称> [--device <cpu|cuda>] [--output <输出视频路径>] [--checkpoint <权重文件路径>] [--input-size <尺寸>] [--fp32] [--streaming]
```

例如：

```powershell
uv run video-to-model data/inputs/<视频文件名>.mp4 --model <模型名称> --output data/outputs/<输出文件名>_depth.mp4
```

PowerShell 中不要原样输入尖括号占位内容；请替换成真实路径或选项值。省略 `--output` 时，默认目标为 `data/outputs/<输入文件名>_depth.mp4`。

## 模型仓库

- 相对深度：Small、Base、Large，分别对应 `vits`、`vitb`、`vitl`。
- 度量深度：使用相同规格并附加 `--metric`。
- 许可可能因仓库和规格而异，请在使用前查看所选 [Hugging Face 模型仓库](https://huggingface.co/depth-anything)。

## 项目结构

```text
models/       模型文件与自动下载的权重
src/          Python 包、CLI、模型注册和推理适配边界
data/inputs/  原始视频
 data/outputs/ 深度生成结果
 docs/         目录架构规划
```

目录规划详见 [docs/directory-structure.md](docs/directory-structure.md)。
