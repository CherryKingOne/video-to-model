# 推理管线资源对照：改造前 vs 改造后

本文件逐项拆解两种实现各自的内存/显存/磁盘/耗时构成，并列出还能继续往下压的手段，供按机器条件取舍。

- 记号：`N` = 抽帧后参与处理的帧数；`h × w` = 解码后的帧分辨率（由 `--max-res` 决定，默认长边 ≤1280）；`H` = 模型输入边长（`--input-size`，默认 518）。
- 每像素字节：RGB uint8 = `3`；float32 = `4`；fp16 = `2`。
- 名词：**RSS** = 进程占用的物理内存峰值（判断 OOM 的指标）；**深度文件** = 落盘的原始深度数值（`.mmap`）；**memmap** = 磁盘文件直接当作 numpy 数组读写，不必全量加载到内存。

## 1. 改造前：上游一次性全量写法

调用链：`read_video_frames()` → `VideoDepthAnything.infer_video_depth()` → `save_video()`。

### 1.1 各阶段常驻内存

| 阶段 | 数据 | 大小 | 生命周期 |
|---|---|---|---|
| 解码 | 整段视频帧（uint8 RGB） | `N·h·w·3` | **全程常驻**。`read_video_frames` 一次性 `np.stack`；调用方 `inference.py` 持有 `frames` 直到函数返回，所以上游内部的 `del frame_list; gc.collect()` 并不能释放它 |
| 窗口输入 | 32 帧 transform 输出（float32） | `32·3·H·H·4` ≈ 102 MB @H=518 | 常数，每窗口重建 |
| 深度累积 | `depth_list`（原分辨率 float32） | `N·h·w·4` | 推理循环内**只增不减** |
| 时序对齐 | `depth_list_aligned` + scale/shift 产生的新数组 | 约再 `N·h·w·4` | 与 `depth_list` **同时存在**（原 list 未释放） |
| 返回 | `np.stack(depth_list[:N])` | 再 `N·h·w·4` | stack 期间三份深度并存 |
| 渲染 | `save_video` 全量求 `min()/max()` | 0（原地扫描） | 但前提是数据必须全在内存 |

### 1.2 峰值公式

```
峰值内存 ≈ N·h·w·(3 + 4 + 4 + 4) = N·h·w·15 字节      （全部为不可回收的堆内存）
```

代入 12,000 帧 @1280×720：`12000 × 921600 × 15 / 1024³ ≈ 154 GB` → 容器/机器内存不足时进程被 OOM Killer 直接 `SIGKILL`（`echo $?` = 137，无任何 Python 报错）。

### 1.3 显存

与 `N` 无关，只取决于窗口：`INFER_LEN=32` 帧 @`--input-size`。`base` @518 fp16 autocast 约几 GB。**这部分从来不是瓶颈。**

## 2. 改造后：常数内存流式管线

调用链：`FrameSource` → `windowed_infer.iter_offline_depths()` → `DepthStore`（memmap）→ `depth_store.render_depth_map()`。结果与第 1 节逐帧等价（`scripts/verify_streaming.py` 对拍，含真权重）。

### 2.1 各阶段内存

| 数据 | 大小 | 是否随 `N` 增长 |
|---|---|---|
| 滑动窗口帧缓冲（uint8） | `32·h·w·3` ≈ 88 MB @720p | ❌ 常数 |
| transform 输出 + `cat` 副本 | `2 × 32·3·H·H·4` ≈ 205 MB @H=518 | ❌ 常数 |
| 单窗口深度（GPU→CPU float32） | `32·h·w·4` ≈ 118 MB @720p | ❌ 常数 |
| 对齐缓冲 `buffer` + `ref_align` | `30·h·w·4 + 2·h·w·4` ≈ 118 MB | ❌ 常数 |
| 写盘缓冲 `WRITE_BLOCK=16` + `np.stack` + fp16 转换 | ≈ 145 MB | ❌ 常数 |
| 渲染分块 `BLOCK_FRAMES=32`：float32 读入 + RGBA 输出 | ≈ 236 MB | ❌ 常数 |
| 基线（Python + torch；GPU 再加 CUDA/cuDNN 上下文） | 约 1.5 GB / +0.5~1 GB | ❌ 常数 |
| memmap 脏页 | `N·h·w·2`(fp16) | ✅ 增长，但是 **file-backed 页缓存，内核可随时丢弃，不参与 OOM 判定** |

即：随长度增长的只有"落盘数据的页缓存"，旧实现那种不可回收的匿名堆内存已经清零。

### 2.2 实测数字

指标含义：

- **进程峰值 RSS**（Resident Set Size，峰值即 `ru_maxrss` / Linux `VmHWM`）：进程实际占用的物理内存最高水位，也就是容器内存限额真正卡的那个量。它包含三部分：torch/Python 基线、本节的常数缓冲、以及 **memmap 被映射进来的页**。前两项不随帧数变，第三项会涨——但它是 file-backed clean page，内核可随时丢弃，**不参与 OOM 判定**。旧实现的 154 GB 全是不可回收的匿名堆页，所以会被 `SIGKILL`。
- **深度文件**：`<输出名>.mmap`，形状 `(frames, height, width)` 的裸二进制原始深度值（默认 fp16 = 每像素 2 字节），配同名 `.json` 记录 `frames/height/width/dtype/fps/depth_min/depth_max`。mp4 只存归一化后的 8bit 伪彩色且经 H.264 有损压缩，**无法反解出深度数值**，所以做 3D/测量/重新渲染必须依赖这个文件。**跑完默认自动删除两者**，要留着就加 `--keep-depth-map`。
- **耗时**：下表为本地 CPU + 随机初始化权重 + `--input-size 168` 的测量，仅用于说明资源结构，不是 GPU 的时间预期。

| 帧数 | 进程峰值 RSS | 深度文件 | 耗时 |
|---|---|---|---|
| 200 | 1.93 GB | 0.34 GB | 35 s |
| 600 | 2.34 GB | 1.03 GB | 100 s |
| 12,000（外推） | ≈2.5~3 GB | 21 GB | ≈13 min（GPU 实测 1.4 s/窗口） |

深度文件大小可手算校验：`600 × 1280 × 720 × 2 B ≈ 1.03 GiB` ✓。200→600 帧 RSS 只涨 0.41 GB 而同期文件涨 0.69 GB，差额就是页缓存记账。

在 Linux 服务器上可以把“可回收”与“不可回收”分开看：

```bash
grep -E "VmHWM|^RssAnon|^RssFile" /proc/$(pgrep -f video-to-model | head -1)/status
# RssAnon：堆内存，不可回收，决定会不会被 OOM kill —— 新实现下应保持恒定
# RssFile：memmap 页缓存，会随帧数增长但内核可随时丢弃
```

读取深度文件：

```python
import json, numpy as np
meta = json.load(open("data/outputs/x_depth.mmap.json"))
depth = np.memmap("data/outputs/x_depth.mmap", dtype=np.float16, mode="r",
                  shape=(meta["frames"], meta["height"], meta["width"]))
```

### 2.3 磁盘需求公式

```
磁盘 ≈ N·h·w·2（fp16，默认） 或 N·h·w·4（fp32）  +  输出 mp4（H.264，默认 CRF 18，可用 `--video-crf` 调小）
```

12,000 帧 @1280×720：fp16 ≈ 21 GB，fp32 ≈ 42 GB。

### 2.4 已实现的采样/降级开关

| 参数 | 影响 |
|---|---|
| `--target-fps` | `N` 线性下降 → 磁盘、耗时同步下降 |
| `--max-res` | 每像素字节数按平方下降 → 磁盘、窗口缓冲、耗时同步下降（同时决定输出分辨率） |
| `--max-len` | 只处理前 N 帧 |
| `--depth-dtype fp16/fp32` | 磁盘 1:2 |
| `--keep-depth-map` | 保留深度文件与元数据（**不加则跑完自动删除**）；自动删除只省长期占用，**不降低过程峰值**（mp4 是从该文件渲染的） |
| `--input-size` | 显存与计算量近似平方下降，几乎不影响 RAM |
| `--video-crf` | 只缩小可视化 mp4（画面流不重编码不影响深度文件）；默认 18，23≈省一半，28≈省三分之二 |
| `--audio`（`true`/`false`） | 把源视频音轨混进输出 mp4：画面流 `-c:v copy` 无损、不重跑推理，内存/磁盘影响可忽略 |
| `--colormap` | 只换可视化色表（`gray` 为传统灰度深度图）；不改变深度数值与任何资源占用 |
| `--streaming` | 逐帧 + temporal cache，窗口缓存恒定；精度下降（d1 0.926→0.836），无跨窗口对齐 |

## 3. 还能继续压的手段

### 3.1 压内存（当前剩余的可压缩量）

| 手段 | 预计收益 | 代价 | 状态 |
|---|---|---|---|
| 写盘时逐帧赋值，去掉 `np.stack(block)` 与 `astype` 副本 | −120~170 MB | 无（不改变任何数值） | 未实现，改动很小 |
| 把 `WRITE_BLOCK` / `BLOCK_FRAMES` 调小 | −100~300 MB | 磁盘写入次数增多，编码吞吐略降 | 常量可调 |
| 降低窗口深度 D2H 的精度（fp16 再回 CPU） | −59 MB/窗口 | **故意不这么做**：对齐的 `scale/shift` 乘法与 `compute_scale_and_shift` 的求和必须跑在 float32 上，否则不再与上游逐帧等价 | ❌ 不建议 |
| 直接输出可视化（不保留深度文件）：跳过 memmap，逐帧归一化写 ffmpeg 管道 | 内存再 −100 MB，**过程磁盘也归零** | 失去原始深度值；且需要放弃"全局 min/max"改为运行区间估计，**明暗对比会变** | 未实现，有语义代价 |

结论：内存已经接近"一个 32 帧窗口"的下限，再压也就是 0.2~0.4 GB 级别。**内存不再是需要优化的对象。**

### 3.2 压磁盘（当前最大的资源项）

| 手段 | 预计收益 | 代价 | 状态 |
|---|---|---|---|
| 默认行为（不加 `--keep-depth-map`） | 长期占用归零（**过程峰值不变**，仍需 ≈21 GB 写入空间） | 不可复渲染、不可复用深度做 3D | ✅ 已实现 |
| `--target-fps 10`（源 30fps） | ÷3 | 时序更稀，快速运动处观感下降 | ✅ 已实现 |
| `--max-res 640` | ÷4 | 深度图/输出分辨率下降 | ✅ 已实现 |
| 深度存半分辨率（`h/2 × w/2`） | ÷4 | 深度细节损失；上采样回原分辨率仍可渲染 | 未实现，可加 `--depth-scale` |
| 压缩存储（分片 zlib `.npz` / 16bit PNG 序列 / EXR） | ÷2~÷4 | CPU 时间增加，读取不再是 `np.memmap` 直接索引 | 未实现 |
| 组合：`--target-fps 10 --max-res 640` | ÷12 → 12k 帧约 **1.8 GB** | 观感/分辨率下降 | ✅ 已实现 |

### 3.3 压显存（只在小卡上爆的时候需要）

1. `--input-size 518 → 420 / 336`（必须是 14 的倍数），显存与耗时近似按平方下降；
2. 保持默认 fp16 autocast（不要加 `--fp32`）；
3. xFormers 已作为依赖装上，memory-efficient attention 生效，省显存明显；
4. `--streaming` 可把窗口开销降到单帧（精度代价见前）。

### 3.4 压耗时

- 抽帧（`--target-fps`）线性见效；`--input-size` 平方见效；`--model video-depth-anything-small` 换小模型。
- **复渲染不要重跑 GPU**：`--render-depth-map <mmap>` 只做归一化+编码，深度只算一次。
- 渲染阶段目前是单进程；如需要可分块并行编码（未实现）。

### 3.5 可视化 mp4 的体积（`--video-crf`）

默认 CRF 18 是“接近视觉无损”档位，而手机/相机产出的源视频往往是 0.5~1 Mbps 的狠压编码，所以深度 mp4 比源视频大是预期行为（源视频未压缩像素约 33 GB，压到 28 MB 是 ≈1200:1；我们用 ≈100:1）。

实测体积比例（600 帧 @1280×720 深度卷）：

| CRF | 体积 | 相对 |
|---|---|---|
| 18（默认） | 1.6 MB | 100% |
| 23 | 871 KB | 54% |
| 28 | 528 KB | 33% |

注意：该测量用的是平移噪声式的合成内容，深度图帧间几乎一致，压缩率远好于真实画面 —— **只能看比例，不能当绝对体积**。深度图的伪彩色色带边界与逐帧抖动会让真实体积明显大于这个估计。

降低 CRF 以外的体积旋钮同样适用于 mp4：`--target-fps`（帧数）与 `--max-res`（像素数）。已有的深度文件可直接重渲染，不必重跑推理。

## 4. 按你的机器的建议

12,000 帧 @1280×720，小显卡 Linux 服务器（CUDA 12.4）：

- 默认命令即可跑通：内存 ≈2.5~3 GB（与长度无关）、磁盘 21 GB、GPU 约 13 分钟。
- **先确认 `/data` 有 ≥25 GB 空闲**。注意这个空间在跑的过程中就需要：mp4 是从深度文件渲染的，默认跑完自动删除只减少长期保留，**不减少过程峰值**。不够就按优先级选一条：
  1. 中转文件换个大盘：`--depth-map /别的盘/scratch/depth.mmap`，跑完自己删；
  2. 要原始深度但可接受稀一点 → `--target-fps 10`（≈7 GB）；
  3. 兼顾 → `--target-fps 10 --max-res 960`（≈4 GB）。
- 只要可视化、不需要数值：什么都不用加（默认就会删掉大文件，只留 mp4）；需要数值做 3D 时加 `--keep-depth-map`。
- 如果显存也吃紧，先降 `--input-size` 到 420，再考虑 `--streaming`。

## 5. 复现这些数字

```bash
uv run python scripts/verify_streaming.py     # 等价性 + 内存倍增检查 + 渲染一致性
```

脚本里包含：`FrameSource` 与 `read_video_frames` 的逐像素对拍、离线/`metric` 分支与上游 `infer_video_depth` 的逐帧对拍（N = 1/5/22/23/32/33/54/55/70）、memmap 往返与统计检查、inferno 映射逐位比对、以及帧数倍增时的 RSS 变化检查。
