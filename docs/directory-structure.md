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
│       ├── inference.py                # 视频解码、推理和结果编码
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

## 数据和模型目录

- `data/inputs/` 存放待处理的原始视频。
- `data/outputs/` 存放生成的深度视频。
- 首次选择模型时，程序从对应 Hugging Face 仓库下载权重到 `models/` 下的模型目录；后续运行会复用本地权重。
- `.gitkeep` 只用于让 Git 保留这些空目录。视频和 `.pth` 权重被 `.gitignore` 排除，不会随代码提交。
