# 视频深度项目目录规划

## 项目目标

本项目基于 [Video Depth Anything](https://github.com/DepthAnything/Depth-Anything-V2) 对视频生成深度结果。模型面向任意长视频，重点关注画质、跨帧时序一致性与泛化能力。相较于基于扩散的方法，它参数量更少、推理速度更快，深度估计的一致性精度更高。

当前阶段先确定工程目录边界；随着后续确定推理加速框架、模型加载方式和运行入口，再细化具体模块。下方结构是规划，不代表这些模块都已实现。

## 建议目录结构

```text
video-to-model/
├── data/
│   ├── inputs/                 # 用户提供的原始视频
│   └── outputs/                # 视频深度生成结果
├── docs/
│   └── directory-structure.md  # 本目录规划与架构演进记录
├── models/                     # 模型权重、配置或模型适配代码（按用途区分）
├── src/
│   └── video_to_model/
│       ├── cli/                # 命令行参数与任务入口
│       ├── config/             # 推理及运行配置
│       ├── video/              # 视频读取、抽帧、预处理与结果编码
│       ├── depth/              # 深度模型封装、权重加载与推理
│       ├── inference/          # 分块/滑窗调度、时序拼接与设备执行
│       └── utils/              # 日志、路径、设备与通用辅助函数
├── tests/                      # 单元与端到端测试
├── .python-version             # uv 使用的 Python 版本提示：3.12
├── pyproject.toml              # 项目元数据、Python 约束与依赖声明
├── uv.lock                     # uv 锁定的依赖版本
└── README.md                   # 项目简介、安装与快速使用说明
```

## 数据目录约定

- `data/inputs/`：存放待处理的原始视频。建议保留原文件名；如需覆盖输出，使用单独的输出路径避免混淆。
- `data/outputs/`：存放由视频生成的深度视频及相关结果。具体文件格式（例如深度可视化视频、逐帧深度数据或元数据）待后续需求确定。
- 数据文件默认不纳入 Git；目录通过 `.gitkeep` 保留。

## 推理架构参考

上游实现基于 PyTorch `nn.Module` 和 `.pth` 权重，没有内置 TensorRT 或 ONNX Runtime 等独立推理引擎。当前可确认的推理特征如下：

- DINOv2 encoder 搭配带 temporal attention 的 `DPTHeadTemporal`。
- 使用 xFormers memory-efficient attention（可用时），并在默认路径中通过 PyTorch autocast 使用 FP16；可切换 FP32。
- 长视频以 32 帧窗口、10 帧重叠进行滑窗推理，避免一次性占用过多显存，再对窗口结果做对齐和拼接。
- 上游提供实验性的 streaming 推理路径；其无需训练但精度有下降，适合作为独立实验能力评估。
- 上游提供 Small、Base、Large 等模型规格。显存与延迟会随模型、精度和输入尺寸变化；官方示例数据基于单张 A100、每窗口 32 帧、518×518 输入，不应直接当作本项目硬件的保证值。

目录职责上，模型加载与上游适配放在 `depth/`，窗口调度与时序融合放在 `inference/`，视频解码及输出编码放在 `video/`。若后续采用 `torch.compile`、TensorRT 或其他加速方案，再在推理层明确后端适配边界和配置，不预设当前已有此类能力。

## 后续细化项

收到后续技术方向后，再据此补充相应目录和职责：推理后端与加速框架、模型权重管理、视频输入输出格式、运行入口、配置方式、日志与错误处理，以及测试范围。
