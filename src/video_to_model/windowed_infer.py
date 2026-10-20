"""Sliding-window depth inference that never accumulates a whole video in memory.

The upstream ``VideoDepthAnything.infer_video_depth`` first stores every decoded frame,
then every raw depth map, and finally runs a second pass over those lists to align the
temporal windows. That is what makes peak memory grow linearly with video length. Here the
same algorithm is expressed as generators: the window padding, the overlap hand-off and the
scale/shift alignment keep only ``INFER_LEN`` frames plus a constant alignment buffer.
"""

from __future__ import annotations

from itertools import islice
from typing import Any, Iterator

import numpy as np

from video_to_model.upstream.video_depth_anything.video_depth import INFER_LEN, INTERP_LEN, KEYFRAMES, OVERLAP

FRAME_STEP = INFER_LEN - OVERLAP


def build_transform(input_size: int, height: int, width: int) -> tuple[Any, int]:
    """Create the upstream input pipeline, including its 16:9 aspect-ratio guard."""
    import cv2
    from torchvision.transforms import Compose

    from video_to_model.upstream.video_depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

    ratio = max(height, width) / min(height, width)
    if ratio > 1.78:  # upstream caps the working resolution for very wide or tall video
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14

    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    return transform, input_size


def iter_frame_windows(frames: Iterator[np.ndarray]) -> Iterator[list[np.ndarray]]:
    """Group a frame stream into upstream's overlapping inference windows.

    Window ``k`` covers sampled frames ``[k * FRAME_STEP, k * FRAME_STEP + INFER_LEN)``, and the
    video is padded with copies of its last frame exactly like upstream does. The window count
    stays identical to ``ceil(length / FRAME_STEP)``, so alignment results are unchanged.
    """
    source = iter(frames)
    tail: list[np.ndarray] = []
    window_index = 0
    frames_read = 0
    while True:
        needed = INFER_LEN if window_index == 0 else FRAME_STEP
        new = list(islice(source, needed))
        frames_read += len(new)
        if FRAME_STEP * window_index >= frames_read:
            break
        window = tail + new
        if len(window) < INFER_LEN:
            window = window + [window[-1].copy()] * (INFER_LEN - len(window))
        yield window
        tail = window[-OVERLAP:]
        window_index += 1


def iter_raw_window_depths(
    model: Any, frames: Iterator[np.ndarray], *, transform: Any, device: str, fp32: bool, height: int, width: int
) -> Iterator[list[np.ndarray]]:
    """Yield the unaligned depth maps of each inference window."""
    import torch
    import torch.nn.functional as F

    pre_input = None
    for window in iter_frame_windows(frames):
        cur_list = [
            torch.from_numpy(transform({"image": frame.astype(np.float32) / 255.0})["image"]).unsqueeze(0).unsqueeze(0)
            for frame in window
        ]
        cur_input = torch.cat(cur_list, dim=1).to(device)
        if pre_input is not None:
            cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(not fp32)):
                depth = model.forward(cur_input)

        depth = depth.to(cur_input.dtype)
        depth = F.interpolate(depth.flatten(0, 1).unsqueeze(1), size=(height, width), mode="bilinear", align_corners=True)
        yield [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]
        pre_input = cur_input


def iter_aligned_depths(raw_windows: Iterator[list[np.ndarray]], *, metric: bool) -> Iterator[np.ndarray]:
    """Align windows on the fly and yield depth frames in final order.

    Only three things have to survive between windows: the ``INTERP_LEN`` output frames that the
    next window is allowed to rewrite, and the two keyframe references used to fit scale/shift.
    """
    from video_to_model.upstream.utils.util import compute_scale_and_shift, get_interpolate_frames

    align_len = OVERLAP - INTERP_LEN
    kf_align_list = KEYFRAMES[:align_len]

    ref_align: list[np.ndarray] = []
    tail: list[np.ndarray] = []

    for window_index, depths in enumerate(raw_windows):
        if window_index == 0:
            ref_align = [depths[kf_id] for kf_id in kf_align_list]
            buffer = list(depths)
        else:
            curr_align = [depths[i] for i in range(len(kf_align_list))]
            if metric:
                scale, shift = 1.0, 0.0
            else:
                scale, shift = compute_scale_and_shift(
                    np.concatenate(curr_align),
                    np.concatenate(ref_align),
                    np.concatenate(np.ones_like(ref_align) == 1),
                )

            post_depth_list = []
            for i in range(align_len, OVERLAP):
                scaled = depths[i] * scale + shift
                scaled[scaled < 0] = 0
                post_depth_list.append(scaled)

            buffer = get_interpolate_frames(tail, post_depth_list)
            for i in range(OVERLAP, INFER_LEN):
                scaled = depths[i] * scale + shift
                scaled[scaled < 0] = 0
                buffer.append(scaled)

            ref_align = ref_align[:1]
            for kf_id in kf_align_list[1:]:
                scaled = depths[kf_id] * scale + shift
                scaled[scaled < 0] = 0
                ref_align.append(scaled)

        # The last INTERP_LEN frames may still be rewritten by the following window.
        flush, tail = buffer[:-INTERP_LEN], buffer[-INTERP_LEN:]
        yield from flush

    yield from tail


def iter_offline_depths(
    model: Any, frames: Iterator[np.ndarray], *, transform: Any, device: str, fp32: bool, height: int, width: int, metric: bool
) -> Iterator[np.ndarray]:
    """Offline (windowed) mode: aligned depth frames streamed one by one."""
    raw = iter_raw_window_depths(model, frames, transform=transform, device=device, fp32=fp32, height=height, width=width)
    yield from iter_aligned_depths(raw, metric=metric)


def iter_streaming_depths(model: Any, frames: Iterator[np.ndarray], *, input_size: int, device: str, fp32: bool) -> Iterator[np.ndarray]:
    """Experimental streaming mode: one frame per pass, temporal caches kept inside the model."""
    for frame in frames:
        yield model.infer_video_depth_one(frame, input_size=input_size, device=device, fp32=fp32)
