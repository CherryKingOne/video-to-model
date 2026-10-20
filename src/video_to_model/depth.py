"""Model adapter boundary for Video Depth Anything."""

from pathlib import Path
from typing import Any

from video_to_model.model_registry import ensure_checkpoint, resolve_model_spec


class ModelNotConfiguredError(RuntimeError):
    """Raised when the upstream model implementation is not installed."""


def build_depth_model(
    *, encoder: str, checkpoint: Path | None, project_root: Path, fp32: bool, streaming: bool, metric: bool = False, device: str = "cpu"
) -> Any:
    """Download if necessary and initialize the upstream Video Depth Anything model."""
    from tqdm import tqdm

    spec = resolve_model_spec(encoder=encoder, metric=metric, project_root=project_root)
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    checkpoint_path = checkpoint or ensure_checkpoint(spec)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    import torch
    if streaming:
        # The upstream streaming implementation takes no ``metric`` argument: it has no
        # cross-window alignment stage, so scale/shift fitting simply does not happen.
        from video_to_model.upstream.video_depth_anything.video_depth_stream import VideoDepthAnything
    else:
        from video_to_model.upstream.video_depth_anything.video_depth import VideoDepthAnything

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The selected checkpoint is available, but PyTorch cannot access a CUDA device. Use --device cpu or install a CUDA-enabled PyTorch build.")
    configs = {
        "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    with tqdm(total=3, desc="加载模型", unit="阶段") as progress:
        if streaming:
            model = VideoDepthAnything(encoder=encoder, **configs[encoder])
        else:
            model = VideoDepthAnything(encoder=encoder, metric=metric, **configs[encoder])
        progress.update(1)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        progress.update(1)
        model = model.to(device).eval()
        progress.update(1)
    return model
