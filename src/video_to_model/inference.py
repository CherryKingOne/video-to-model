"""Video decoding, depth inference and result encoding orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from video_to_model.frame_source import FrameSource

# Depth frames buffered before they are pushed to the memory-mapped volume.
WRITE_BLOCK = 16


def default_depth_map_path(output_path: Path) -> Path:
    """Location of the raw depth volume that backs the rendered video."""
    return output_path.with_name(f"{output_path.stem}.mmap")


def move_result_to_cpu(value: Any) -> Any:
    """Detach a tensor-like result and place it in CPU memory."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value


def run_video_inference(
    *,
    model: Any,
    input_path: Path,
    output_path: Path,
    input_size: int,
    fp32: bool,
    streaming: bool,
    device: str = "cpu",
    max_len: int = -1,
    target_fps: float = -1,
    max_res: int = 1280,
    depth_map: Path | None = None,
    depth_dtype: str = "fp16",
    keep_depth_map: bool = True,
) -> None:
    """Stream a video through the model, store depths on disk, then encode the visualization.

    Memory stays bounded by one inference window: frames are decoded lazily and each depth map is
    written to a memory-mapped volume as soon as it is produced, instead of being collected in a
    list. The mp4 is rendered afterwards in two streaming passes over that file.
    """
    import torch
    from tqdm import tqdm

    from video_to_model import depth_store, windowed_infer

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but no CUDA device is available.")
    # CPU autocast to fp16 is unsupported on many CPU builds; use fp32 there.
    fp32 = fp32 or device == "cpu"

    source = FrameSource(input_path, target_fps=target_fps, max_res=max_res, max_len=max_len)
    info = source.info
    if info.frame_count == 0:
        raise ValueError(f"No frames could be decoded from {input_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_path = Path(depth_map) if depth_map else default_depth_map_path(output_path)

    frames = source.counted()
    if streaming:
        used_input_size = input_size
        depths = windowed_infer.iter_streaming_depths(model, frames, input_size=input_size, device=device, fp32=fp32)
    else:
        transform, used_input_size = windowed_infer.build_transform(input_size, info.height, info.width)
        depths = windowed_infer.iter_offline_depths(
            model,
            frames,
            transform=transform,
            device=device,
            fp32=fp32,
            height=info.height,
            width=info.width,
            metric=bool(getattr(model, "metric", False)),
        )

    store = depth_store.DepthStore(map_path, height=info.height, width=info.width, dtype=depth_dtype, capacity=info.frame_count)
    progress = tqdm(total=info.frame_count, desc="推理深度", unit="帧")
    buffer: list[np.ndarray] = []
    try:
        for depth in depths:
            buffer.append(depth)
            if len(buffer) >= WRITE_BLOCK:
                store.append(buffer)
                progress.update(len(buffer))
                buffer = []
        if buffer:
            store.append(buffer)
            progress.update(len(buffer))
    finally:
        progress.close()

    # Window padding produces a few trailing frames beyond the source length.
    store.trim_to(source.frames_read)
    store.finalize(
        meta={
            "source": str(input_path),
            "fps": info.fps,
            "input_size": used_input_size,
            "device": device,
            "streaming": streaming,
            "model": type(model).__name__,
        }
    )

    depth_store.render_depth_map(map_path, output_path)

    if store.overflow_frames:
        print(
            f"警告：{store.overflow_frames} 帧深度值超出 fp16 表示范围并被记为 inf（这些帧不参与全局明暗统计）。"
            "若要精确深度值请用 --depth-dtype fp32 重跑。"
        )

    if keep_depth_map:
        print(f"深度原始值已保存: {map_path}（可用 --render-depth-map {map_path} 重新渲染）")
    else:
        map_path.unlink(missing_ok=True)
        depth_store.metadata_path(map_path).unlink(missing_ok=True)
