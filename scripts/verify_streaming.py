"""Regression checks for the streaming/chunked depth pipeline.

Run with:  uv run python scripts/verify_streaming.py

The important assertion is that the reworked pipeline produces *exactly* the same depth
frames as the upstream ``VideoDepthAnything.infer_video_depth`` it replaced, for every
window-boundary edge case, and that the visualization matches ``dc_utils.save_video``.
"""

import resource
import sys
from pathlib import Path

import numpy as np
import torch

import imageio
from matplotlib import colormaps

from video_to_model import depth_store, windowed_infer
from video_to_model.frame_source import FrameSource
from video_to_model.upstream.utils.dc_utils import read_video_frames, save_video
from video_to_model.upstream.video_depth_anything.video_depth import VideoDepthAnything

WORK = Path("/tmp/vtm-check")
WORK.mkdir(parents=True, exist_ok=True)


class FakeDepth(VideoDepthAnything):
    """Same module surface as the real model, but a deterministic cheap forward."""

    def __init__(self, metric: bool = False) -> None:
        torch.nn.Module.__init__(self)
        self.metric = metric

    def forward(self, x):
        length = x.shape[1]
        base = x.mean(dim=2)
        ramp = torch.arange(length, device=x.device, dtype=base.dtype).view(1, length, 1, 1)
        return torch.relu(base * (1.0 + 0.37 * torch.sin(ramp * 0.11)) + 0.05 * ramp)


def make_video(path: Path, frames: int, height: int, width: int, fps: int = 30) -> None:
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1, codec="libx264")
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    try:
        for index in range(frames):
            writer.append_data(np.clip(base.astype(np.int16) + index, 0, 255).astype(np.uint8))
    finally:
        writer.close()


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label} {extra}")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    video = WORK / "src_70.mp4"
    if not video.is_file():
        make_video(video, 70, 120, 160)

    # --- lazy decoding must sample exactly like the upstream helper
    for label, options in (
        ("defaults", {"max_len": -1, "target_fps": -1, "max_res": 1280}),
        ("max_len=40", {"max_len": 40, "target_fps": -1, "max_res": 1280}),
        ("target_fps=10", {"max_len": -1, "target_fps": 10, "max_res": 1280}),
        ("max_res=100", {"max_len": -1, "target_fps": -1, "max_res": 100}),
        ("max_len=50,fps=12,res=90", {"max_len": 50, "target_fps": 12, "max_res": 90}),
    ):
        reference, ref_fps = read_video_frames(str(video), options["max_len"], options["target_fps"], options["max_res"])
        source = FrameSource(video, **options)
        streamed = np.stack(list(source.frames()))
        check(
            f"FrameSource 与上游 read_video_frames 逐像素一致: {label}",
            streamed.shape == reference.shape and np.array_equal(streamed, reference) and source.info.fps == float(ref_fps),
            f"shape={streamed.shape} fps={source.info.fps}",
        )

    model = FakeDepth(metric=False)
    frames_all, _ = read_video_frames(str(video), -1, -1, 1280)
    height, width = frames_all.shape[1:3]

    # --- window grouping + temporal alignment must be bit-identical to upstream
    for count in (1, 5, 22, 23, 32, 33, 54, 55, 70):
        frames = frames_all[:count]
        reference, _ = model.infer_video_depth(frames, 30, input_size=42, device="cpu", fp32=True)
        transform, _ = windowed_infer.build_transform(42, height, width)
        ours = np.stack(
            list(
                windowed_infer.iter_offline_depths(
                    model,
                    iter(frames),
                    transform=transform,
                    device="cpu",
                    fp32=True,
                    height=height,
                    width=width,
                    metric=False,
                )
            )
        )[:count]
        check(
            f"离线模式深度结果与上游完全相同: N={count}",
            ours.shape == reference.shape and np.array_equal(ours, reference),
            f"shape={ours.shape}",
        )

    metric_model = FakeDepth(metric=True)
    reference, _ = metric_model.infer_video_depth(frames_all, 30, input_size=42, device="cpu", fp32=True)
    transform, _ = windowed_infer.build_transform(42, height, width)
    ours = np.stack(
        list(
            windowed_infer.iter_offline_depths(
                metric_model,
                iter(frames_all),
                transform=transform,
                device="cpu",
                fp32=True,
                height=height,
                width=width,
                metric=True,
            )
        )
    )[: len(frames_all)]
    check("metric 分支与上游完全相同", np.array_equal(ours, reference))

    # --- depth volume round-trip
    for dtype, tolerance in (("fp32", {"rtol": 0, "atol": 0}), ("fp16", {"rtol": 2e-3, "atol": 2e-3})):
        path = WORK / f"depth_{dtype}.mmap"
        store = depth_store.DepthStore(path, height=height, width=width, dtype=dtype, capacity=8)
        for start in range(0, len(ours), 7):
            store.append(list(ours[start : start + 7]))
        check(f"{dtype}: 超出初始 capacity 后仍可增长", store.count >= len(ours) - 7)
        store.trim_to(len(ours))
        store.finalize(meta={"source": str(video), "fps": 30, "input_size": 42, "device": "cpu", "streaming": False, "model": "FakeDepth"})
        back = np.concatenate(list(depth_store.iter_depth_blocks(path, block_frames=5)), axis=0)
        check(f"{dtype}: memmap 往返读写一致", back.shape == ours.shape and np.allclose(back, ours, **tolerance), f"shape={back.shape}")
        low, high = depth_store.global_min_max(path)
        # 统计值本身也存不了超过该 dtype 的精度：fp16 在 4.34 附近的格点间距约为 0.0039，
        # 因此阈值必须按 dtype 的机器精度缩放，而不是固定绝对值。
        tol = np.finfo(depth_store.DTYPES[dtype]).eps * max(abs(low), abs(high), 1.0)
        check(
            f"{dtype}: 分块统计等于整块统计",
            abs(low - float(ours.min())) <= tol and abs(high - float(ours.max())) <= tol,
            f"range=({low:.4f}, {high:.4f}) 允许误差={tol:.2e}",
        )
        meta = depth_store.load_metadata(path)
        check(f"{dtype}: sidecar 记录帧数与 dtype", meta["frames"] == len(ours) and meta["dtype"] == dtype)

    # --- visualization must match upstream's formula, then its mp4
    lut = np.array(colormaps["inferno"].colors)
    d_min, d_max = float(ours.min()), float(ours.max())
    expected = (lut[((ours - d_min) / (d_max - d_min) * 255).astype(np.uint8)] * 255).astype(np.uint8)
    rendered = np.concatenate(
        [depth_store.visualize(block, d_min, d_max, lut) for block in depth_store.iter_depth_blocks(WORK / "depth_fp32.mmap", block_frames=9)],
        axis=0,
    )
    check("inferno 映射与上游 save_video 逐位一致", np.array_equal(rendered, expected))

    depth_store.render_depth_map(WORK / "depth_fp32.mmap", WORK / "ours.mp4")
    save_video(ours, str(WORK / "upstream.mp4"), fps=30, is_depths=True)
    ours_video = read_video_frames(str(WORK / "ours.mp4"), -1, -1, -1)[0]
    upstream_video = read_video_frames(str(WORK / "upstream.mp4"), -1, -1, -1)[0]
    check("两条渲染路径帧数一致", ours_video.shape == upstream_video.shape, f"{ours_video.shape}")
    diff = np.abs(ours_video.astype(np.int16) - upstream_video.astype(np.int16)).mean()
    check("渲染出的 mp4 与上游 save_video 视觉一致", diff < 6.0, f"平均像素差={diff:.3f}")
    check(
        "渲染后把全局统计写回 sidecar",
        "depth_min" in depth_store.metadata_path(WORK / "depth_fp32.mmap").read_text(encoding="utf-8"),
    )

    # --- memory behaviour: the pipeline heap stays flat while the file grows
    peaks: list[tuple[int, float, int]] = []
    for count in (200, 1200):
        big = WORK / f"src_{count}.mp4"
        if not big.is_file():
            make_video(big, count, 120, 160)
        source = FrameSource(big, max_res=1280)
        transform, _ = windowed_infer.build_transform(42, source.info.height, source.info.width)
        store = depth_store.DepthStore(WORK / f"big_{count}.mmap", height=source.info.height, width=source.info.width, dtype="fp16", capacity=64)
        buffer = []
        for depth in windowed_infer.iter_offline_depths(
            model,
            source.counted(),
            transform=transform,
            device="cpu",
            fp32=True,
            height=source.info.height,
            width=source.info.width,
            metric=False,
        ):
            buffer.append(depth)
            if len(buffer) >= 16:
                store.append(buffer)
                buffer = []
        if buffer:
            store.append(buffer)
        store.trim_to(source.frames_read)
        store.finalize(meta={"source": str(big), "fps": source.info.fps, "input_size": 42, "device": "cpu", "streaming": False, "model": "FakeDepth"})
        # macOS reports bytes, Linux reports kibibytes.
        peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_kib = peak_kib / 1024 if sys.platform == "darwin" else peak_kib
        peaks.append((source.frames_read, peak_kib / 1024**2, store.overflow_frames))

    for frames, peak, overflow in peaks:
        size = Path(WORK / f"big_{frames}.mmap").stat().st_size / 1024**2
        print(f"     N={frames:>5} 帧：进程峰值 RSS≈{peak:.2f} GB，落盘 {size:.1f} MB，fp16 溢出帧={overflow}")
    growth = peaks[1][0] / peaks[0][0]
    delta = peaks[1][1] - peaks[0][1]
    check("帧数倍增时进程峰值 RSS 基本持平", abs(delta) < 0.25, f"帧数 ×{growth:.0f}，RSS 变化 {delta:+.2f} GB")

    # Analytic comparison for the real failure case: 12k frames at 1280x720.
    frames, height, width = 12000, 720, 1280
    upstream_gb = frames * height * width * 15 / 1024**3
    print(
        f"     参照：{frames} 帧 @{width}x{height} 上游需常驻约 {upstream_gb:.0f} GB 内存；"
        f"流式管线只需常数内存 + {frames * height * width * 2 / 1024**3:.0f} GB 磁盘(fp16)"
    )

    print("\n全部检查通过")


if __name__ == "__main__":
    main()
