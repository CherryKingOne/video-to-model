"""Download and locate Video Depth Anything checkpoints from the Hub."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

# Hugging Face defaults both metadata and file-request timeouts to 10 seconds.
# Set longer defaults before importing huggingface_hub; user-provided values win.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    directory: Path
    checkpoint_name: str


MODEL_VARIANTS = {
    (False, "vits"): ("depth-anything/Video-Depth-Anything-Small", "Video-Depth-Anything-Small", "video_depth_anything_vits.pth"),
    (False, "vitb"): ("depth-anything/Video-Depth-Anything-Base", "Video-Depth-Anything-Base", "video_depth_anything_vitb.pth"),
    (False, "vitl"): ("depth-anything/Video-Depth-Anything-Large", "Video-Depth-Anything-Large", "video_depth_anything_vitl.pth"),
    (True, "vits"): ("depth-anything/Metric-Video-Depth-Anything-Small", "Metric-Video-Depth-Anything-Small", "metric_video_depth_anything_vits.pth"),
    (True, "vitb"): ("depth-anything/Metric-Video-Depth-Anything-Base", "Metric-Video-Depth-Anything-Base", "metric_video_depth_anything_vitb.pth"),
    (True, "vitl"): ("depth-anything/Metric-Video-Depth-Anything-Large", "Metric-Video-Depth-Anything-Large", "metric_video_depth_anything_vitl.pth"),
}


def resolve_model_spec(*, encoder: str, metric: bool, project_root: Path) -> ModelSpec:
    try:
        repo_id, folder, filename = MODEL_VARIANTS[(metric, encoder)]
    except KeyError as exc:
        raise ValueError(f"Unsupported model variant: encoder={encoder}, metric={metric}") from exc
    category = "Metric-Depth-Model" if metric else "Relative-Depth—Model"
    return ModelSpec(repo_id, project_root / "models" / category / folder, filename)


def ensure_checkpoint(spec: ModelSpec) -> Path:
    """Return the local checkpoint, downloading it into the requested models folder if absent."""
    checkpoint = spec.directory / spec.checkpoint_name
    if checkpoint.is_file():
        return checkpoint

    spec.directory.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            downloaded = snapshot_download(
                repo_id=spec.repo_id,
                local_dir=spec.directory,
                allow_patterns=[spec.checkpoint_name],
                etag_timeout=60,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                delay = attempt * 3
                print(f"Hugging Face connection failed (attempt {attempt}/3): {exc}. Retrying in {delay}s...")
                time.sleep(delay)
    else:
        raise RuntimeError(
            f"Could not reach Hugging Face to download {spec.repo_id} after 3 attempts. "
            "The checkpoint has not been downloaded. Check network/proxy access, or configure "
            "HF_ENDPOINT to a Hub endpoint reachable from this environment. "
            f"Last error: {last_error}"
        ) from last_error

    checkpoint = Path(downloaded) / spec.checkpoint_name
    if not checkpoint.is_file():
        raise RuntimeError(
            f"Repository {spec.repo_id} did not provide expected checkpoint "
            f"{spec.checkpoint_name}. Inspect the repository files and update MODEL_VARIANTS."
        )
    return checkpoint
