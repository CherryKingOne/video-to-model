# 项目目录结构

本项目使用 Video Depth Anything 对视频进行深度估计。当前代码已实现命令行入口、模型下载与加载、视频推理和结果输出；本文件记录实际目录结构，避免把尚未使用的空目录误认为缺失的代码。

## 实际结构

```text
video-to-model/
├── data/
│   ├── inputs/                         # 原始视频；媒体文件不提交到 Git
│   │   └── .gitkeep
│   └── outputs/                        # 深度结果；生成文件不提交到 Git
│       └── .gitkeep
├── docs/
│   └── directory-structure.md
├── scripts/
│   └── verify_streaming.py                 # 与上游逐帧对拍的开发验证脚本
├── models/                             # 模型权重下载位置；权重不提交到 Git
│   ├── Metric-Depth-Model/
│   │   ├── Metric-Video-Depth-Anything-Small/.gitkeep
│   │   ├── Metric-Video-Depth-Anything-Base/.gitkeep
│   │   └── Metric-Video-Depth-Anything-Large/.gitkeep
│   └── Relative-Depth—Model/
│       ├── Video-Depth-Anything-Small/.gitkeep
│       ├── Video-Depth-Anything-Base/.gitkeep
│       └── Video-Depth-Anything-Large/.gitkeep
├── src/
│   └── video_to_model/
│       ├── cli.py                      # 命令行参数与入口
│       ├── config.py                   # 项目路径配置
│       ├── depth.py                    # 模型创建、下载和权重加载
│       ├── frame_source.py             # 按需逐帧解码（抽帧/限长/限分辨率）
│       ├── windowed_infer.py           # 滑窗推理与常数内存时序对齐
│       ├── depth_store.py              # 深度 memmap 落盘、全局统计与流式渲染
│       ├── inference.py                # 解码→推理→落盘→渲染的编排
│       ├── model_registry.py           # Hugging Face 模型映射
│       └── upstream/                   # Video Depth Anything 上游模型实现
├── .python-version                     # Python 3.12
├── pyproject.toml                      # 项目元数据和依赖
├── uv.lock                             # 锁定依赖版本
└── README.md                           # 安装与运行说明
```

## 源码模块说明

`src/video_to_model/` 下的 `cli.py`、`config.py`、`depth.py` 和 `inference.py` 是标准 Python 模块，直接放在包目录中是正确的布局，不需要再各自套一层同名目录。早期架构草案中的 `cli/`、`config/`、`video/` 等空子目录已移除。

上游模型代码放在 `src/video_to_model/upstream/`，应用自己的 CLI、模型仓库映射与视频处理代码则放在包根目录，便于区分上游实现和本项目的适配层。

## 内存分层的模块边界

上游的 `VideoDepthAnything.infer_video_depth` 会一次性持有整段视频和全部深度图，峰值内存随帧数线性增长。本项目的适配层不再调用它，而是复用同一个模型的 `forward`，在适配层里重写出语义完全相同的分批流程：

- `frame_source.py`：等价于 `dc_utils.read_video_frames` 的采样规则，但按需逐帧产出。
- `windowed_infer.py`：等价于上游的滑窗 + scale/shift 对齐，但只保留滑窗缓冲与对齐所需的常数帧。
- `depth_store.py`：深度写入 memmap，并在渲染时分块求全局明暗范围（上游需要全量数组才能求）。

`upstream/` 下的算法代码保持原样，`scripts/verify_streaming.py` 用同一个假模型对两条路径逐帧对拍，保证分批改写没有改变结果。唯一例外是 `upstream/utils/dc_utils.py`：matplotlib 3.11 移除了 `cm.get_cmap`，已改为 `colormaps[...]` 兼容写法，否则任何成功的运行都会在编码阶段报错。

## 数据和模型目录

- `data/inputs/` 存放待处理的原始视频。
- `data/outputs/` 存放生成的深度视频，以及默认保留的原始深度 memmap（`<输出名>.mmap` + 同名 `.json` 元数据）。
- 首次选择模型时，程序从对应 Hugging Face 仓库下载权重到 `models/` 下的模型目录；后续运行会复用本地权重。
- `.gitkeep` 只用于让 Git 保留这些空目录。视频和 `.pth` 权重被 `.gitignore` 排除，不会随代码提交。
