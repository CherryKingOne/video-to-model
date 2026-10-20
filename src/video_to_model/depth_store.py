"""Write per-frame depths straight to disk and render them without loading a video."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

DTYPES = {"fp16": np.float16, "fp32": np.float32}

# Frames read per pass when scanning or encoding; keeps render memory independent of length.
BLOCK_FRAMES = 32


def metadata_path(depth_map: Path) -> Path:
    """Sidecar describing how the raw volume was written."""
    return Path(str(depth_map) + ".json")


def load_metadata(depth_map: Path) -> dict[str, Any]:
    sidecar = metadata_path(depth_map)
    if not sidecar.is_file():
        raise ValueError(f"Missing depth-map metadata: {sidecar}")
    return json.loads(sidecar.read_text(encoding="utf-8"))


class DepthStore:
    """Append-only ``numpy`` memmap holding a ``(frames, height, width)`` depth volume.

    The file grows in power-of-two blocks so the total frame count does not have to be
    known upfront, and it is trimmed to the exact size on :meth:`finalize`.
    """

    def __init__(self, path: Path, *, height: int, width: int, dtype: str = "fp16", capacity: int | None = None) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported depth dtype: {dtype}")
        self.path = Path(path)
        self.height = height
        self.width = width
        self.dtype_name = dtype
        self.dtype = DTYPES[dtype]
        self.rows = height * width
        self.count = 0
        self.overflow_frames = 0  # frames holding values outside the fp16 range
        self._capacity = max(int(capacity or 0), 64)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open(self._capacity)

    def _open(self, capacity: int) -> None:
        self._capacity = capacity
        with open(self.path, "wb") as handle:
            handle.truncate(capacity * self.rows * self.dtype().itemsize)
        self._array = np.memmap(self.path, dtype=self.dtype, mode="r+", shape=(capacity, self.height, self.width))

    def _grow(self, needed: int) -> None:
        """Reopen the volume with a larger capacity; unwritten space stays sparse on disk."""
        self._array.flush()
        del self._array
        capacity = self._capacity
        while needed > capacity:
            capacity *= 2
        with open(self.path, "r+b") as handle:
            handle.truncate(capacity * self.rows * self.dtype().itemsize)
        self._capacity = capacity
        self._array = np.memmap(self.path, dtype=self.dtype, mode="r+", shape=(capacity, self.height, self.width))

    def append(self, frames: np.ndarray | list[np.ndarray]) -> None:
        """Write depth frames at the current end of the volume."""
        block = np.stack(frames) if isinstance(frames, list) else np.asarray(frames)
        if block.ndim != 3 or block.shape[1:] != (self.height, self.width):
            raise ValueError(f"Expected frames of shape (n, {self.height}, {self.width}), got {block.shape}")
        if self.count + len(block) > self._capacity:
            self._grow(self.count + len(block))
        cast = block.astype(self.dtype, copy=False)
        if self.dtype is np.float16:
            # A badly conditioned scale/shift fit can produce depths past the fp16 range; they
            # become inf, are skipped by the statistics pass, and must be reported.
            overflow = ~np.isfinite(cast)
            self.overflow_frames += int(np.count_nonzero(overflow.any(axis=(1, 2))))
        self._array[self.count : self.count + len(block)] = cast
        self.count += len(block)

    def flush(self) -> None:
        self._array.flush()

    def trim_to(self, frames: int) -> None:
        """Forget rows past ``frames``; window padding generates a few extra depth maps."""
        if frames < self.count:
            self.count = frames

    def finalize(self, *, meta: dict[str, Any]) -> Path:
        """Trim the file to the written frames, record metadata and return the map path."""
        self._array.flush()
        del self._array
        os.truncate(self.path, self.count * self.rows * self.dtype().itemsize)
        payload = dict(meta)
        payload.update(
            {
                "frames": self.count,
                "height": self.height,
                "width": self.width,
                "dtype": self.dtype_name,
                "fp16_overflow_frames": self.overflow_frames,
            }
        )
        metadata_path(self.path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.path


def iter_depth_blocks(depth_map: Path, *, meta: dict[str, Any] | None = None, block_frames: int = BLOCK_FRAMES) -> Iterator[np.ndarray]:
    """Stream the volume in blocks; memory stays at ``block_frames`` regardless of length."""
    meta = meta or load_metadata(depth_map)
    dtype = DTYPES[meta["dtype"]]
    frames, height, width = int(meta["frames"]), int(meta["height"]), int(meta["width"])
    array = np.memmap(depth_map, dtype=dtype, mode="r", shape=(frames, height, width))
    try:
        for start in range(0, frames, block_frames):
            yield np.asarray(array[start : start + block_frames], dtype=np.float32)
    finally:
        del array


def global_min_max(depth_map: Path, *, meta: dict[str, Any] | None = None, progress: bool = True) -> tuple[float, float]:
    """Return cached or recomputed depth range, ignoring non-finite values."""
    from tqdm import tqdm

    meta = meta or load_metadata(depth_map)
    cached = meta.get("depth_min"), meta.get("depth_max")
    if cached[0] is not None and cached[1] is not None:
        return float(cached[0]), float(cached[1])

    low, high = float("inf"), float("-inf")
    with tqdm(total=int(meta["frames"]), desc="统计明暗范围", unit="帧", disable=not progress) as bar:
        for block in iter_depth_blocks(depth_map, meta=meta):
            finite = block[np.isfinite(block)]
            if finite.size:
                low = min(low, float(finite.min()))
                high = max(high, float(finite.max()))
            bar.update(len(block))
    if low == float("inf") or high == float("-inf"):
        low, high = 0.0, 1.0
    return low, high


def visualize(block: np.ndarray, depth_min: float, depth_max: float, lut: np.ndarray) -> np.ndarray:
    """Apply upstream's inferno mapping to a block of float32 depths."""
    span = depth_max - depth_min
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    normalized = np.where(np.isfinite(block), block, depth_min)
    depth_norm = ((normalized - depth_min) / span * 255).astype(np.uint8)
    return (lut[depth_norm] * 255).astype(np.uint8)


def update_metadata(depth_map: Path, **updates: Any) -> dict[str, Any]:
    """Merge ``updates`` into the sidecar so a later render can skip the statistics pass."""
    meta = load_metadata(depth_map)
    meta.update(updates)
    metadata_path(depth_map).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def ffmpeg_executable() -> str:
    """Path to the ffmpeg binary bundled with imageio-ffmpeg (no system install needed)."""
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def has_audio_stream(video: Path) -> bool:
    """Detect an audio track by reading ffmpeg's stream listing."""
    import subprocess

    listing = subprocess.run([ffmpeg_executable(), "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    return "Audio:" in listing.stderr


def mux_audio(video: Path, audio_source: Path) -> bool:
    """Add the first audio track of ``audio_source`` to ``video`` in place.

    The depth picture stream is copied untouched, so muxing costs no quality and no re-encode.
    Returns ``False`` when the source carries no audio at all.
    """
    import subprocess

    if not has_audio_stream(audio_source):
        return False
    temporary = video.with_name(f"{video.stem}.audio.mp4")
    subprocess.run(
        [
            ffmpeg_executable(),
            "-y",
            "-i", str(video),
            "-i", str(audio_source),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(temporary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    temporary.replace(video)
    return True


def color_lut(colormap: str) -> np.ndarray:
    """Sample a matplotlib colormap into a 256-entry lookup table.

    Discrete maps (``inferno``) expose their table directly, which keeps upstream's exact
    colours; continuous maps (``gray``, ``hot``) have no ``.colors`` and must be sampled.
    """
    from matplotlib import colormaps

    try:
        return np.array(colormaps[colormap].colors)
    except AttributeError:
        return np.array(colormaps[colormap](np.linspace(0.0, 1.0, 256)))


def render_depth_map(
    depth_map: Path,
    output_video: Path,
    *,
    fps: float | None = None,
    colormap: str = "inferno",
    crf: int = 18,
    audio_source: Path | None = None,
) -> bool | None:
    """Encode a depth volume to an mp4 in two streaming passes over the file.

    Returns ``None`` when no audio was requested, otherwise whether a track was muxed in.
    """
    import imageio
    from tqdm import tqdm

    meta = load_metadata(depth_map)
    cached_range = meta.get("depth_min") is not None
    depth_min, depth_max = global_min_max(depth_map, meta=meta)
    if not cached_range:
        update_metadata(depth_map, depth_min=depth_min, depth_max=depth_max)
    lut = color_lut(colormap)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(output_video),
        fps=fps if fps is not None else float(meta["fps"]),
        macro_block_size=1,
        codec="libx264",
        ffmpeg_params=["-crf", str(crf)],
    )
    try:
        with tqdm(total=int(meta["frames"]), desc="编码视频", unit="帧") as bar:
            for block in iter_depth_blocks(depth_map, meta=meta):
                for frame in visualize(block, depth_min, depth_max, lut):
                    writer.append_data(frame)
                bar.update(len(block))
    finally:
        writer.close()

    if audio_source is None:
        return None
    return mux_audio(output_video, Path(audio_source))
