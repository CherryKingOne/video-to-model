"""Video decoding and depth-video encoding adapters."""

from pathlib import Path
from typing import Any


def move_result_to_cpu(value: Any) -> Any:
    """Detach a tensor-like result and place it in CPU memory."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value


def run_video_inference(
    *, model: Any, input_path: Path, output_path: Path, input_size: int, fp32: bool, streaming: bool, device: str = "cpu"
) -> None:
    """Decode an input video, infer depth, and encode the visualization video."""
    import torch
    from video_to_model.upstream.utils.dc_utils import read_video_frames, save_video

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but no CUDA device is available.")
    frames, fps = read_video_frames(str(input_path), -1, -1, 1280)
    if len(frames) == 0:
        raise ValueError(f"No frames could be decoded from {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # CPU autocast to fp16 is unsupported on many CPU builds; use fp32 there.
    depths, fps = model.infer_video_depth(
        frames,
        fps,
        input_size=input_size,
        device=device,
        fp32=(fp32 or device == "cpu"),
    )
    save_video(depths, str(output_path), fps=fps, is_depths=True)
