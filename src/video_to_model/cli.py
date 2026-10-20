"""Command line interface for video depth generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from video_to_model.config import ProjectPaths
from video_to_model.depth import ModelNotConfiguredError, build_depth_model
from video_to_model.inference import run_video_inference

MODEL_NAMES = (
    "video-depth-anything-small",
    "video-depth-anything-base",
    "video-depth-anything-large",
    "metric-video-depth-anything-small",
    "metric-video-depth-anything-base",
    "metric-video-depth-anything-large",
)

# 可视化配色。上游固定用 inferno（热图风格），gray 为传统灰度深度图。
# 映射规则是“深度值越大越亮”，与色表无关，所以换色表不改变任何数值。
COLORMAPS = ("inferno", "gray", "gray_r", "viridis", "magma", "plasma", "turbo", "hot")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-to-model",
        description="Generate a depth video from an input video.",
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input video path")
    parser.add_argument("--output", type=Path, help="Output depth video path")
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        help="Model repository variant; for example video-depth-anything-base",
    )
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl"), default="vits")
    parser.add_argument("--metric", action="store_true", help="Use metric-depth checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint (.pth)")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Inference device (default: cpu)")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 autocast")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Experimental per-frame streaming mode (temporal cache instead of window alignment; lower accuracy)",
    )
    sampling = parser.add_argument_group("video sampling")
    sampling.add_argument("--max-len", type=int, default=-1, help="Stop after decoding this many frames; -1 means no limit")
    sampling.add_argument("--target-fps", type=float, default=-1, help="Target sampling fps; -1 keeps the source fps (default: -1)")
    sampling.add_argument("--max-res", type=int, default=1280, help="Cap the decoded frame's long side in pixels (default: 1280)")
    storage = parser.add_argument_group("depth output")
    storage.add_argument("--depth-map", type=Path, help="Path for the raw depth volume (default: <output>.stem.mmap)")
    storage.add_argument("--depth-dtype", choices=("fp16", "fp32"), default="fp16", help="Depth volume precision (default: fp16)")
    storage.add_argument(
        "--discard-depth-map",
        action="store_true",
        help="Delete the raw depth volume after rendering it (default: keep it)",
    )
    storage.add_argument(
        "--render-depth-map", type=Path, help="Only re-render this existing depth volume into a video"
    )
    storage.add_argument(
        "--video-crf",
        type=int,
        default=18,
        help="H.264 quality for the rendered video; larger means smaller files (default 18, near-lossless; 23~28 saves space)",
    )
    storage.add_argument(
        "--audio",
        nargs="?",
        const="true",
        default="false",
        choices=("true", "false"),
        help="是否把源视频音轨写进输出视频：--audio 或 --audio true 开启，--audio false 关闭（默认关闭）",
    )
    storage.add_argument(
        "--colormap",
        default="inferno",
        choices=COLORMAPS,
        help="可视化配色；gray 输出灰度深度视频，gray_r 明暗反向（默认 inferno）",
    )
    return parser


def _resolve(path: Path | None, root: Path) -> Path | None:
    return None if path is None else (path if path.is_absolute() else root / path)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    paths = ProjectPaths.from_project_root(Path.cwd())

    if args.model:
        metric = args.model.startswith("metric-")
        encoder = {"small": "vits", "base": "vitb", "large": "vitl"}[args.model.rsplit("-", 1)[1]]
        if args.metric or args.encoder != "vits":
            parser.error("Use either --model or the legacy --encoder/--metric options, not both")
    else:
        metric = args.metric
        encoder = args.encoder

    try:
        if args.render_depth_map:
            map_path = _resolve(args.render_depth_map, paths.root)
            if not map_path.is_file():
                parser.error(f"Depth map does not exist: {map_path}")
            from video_to_model.depth_store import load_metadata, render_depth_map

            output_path = _resolve(args.output, paths.root) or map_path.with_suffix(".mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            audio_source = None
            if args.audio == "true":
                source = load_metadata(map_path).get("source")
                if not source or not Path(source).is_file():
                    parser.error(f"--audio 需要深度文件记录的源视频可访问，但找不到: {source}")
                audio_source = Path(source)
            muxed = render_depth_map(map_path, output_path, crf=args.video_crf, colormap=args.colormap, audio_source=audio_source)
            if muxed is False:
                print(f"源视频没有音轨，输出保持静音: {audio_source}")
            print(f"已重新渲染: {output_path}{'（已带上音轨）' if muxed else ''}")
            return 0

        if args.input is None:
            parser.error("input video is required (or use --render-depth-map to only re-render)")

        input_path = _resolve(args.input, paths.root)
        if not input_path.is_file():
            parser.error(f"Input video does not exist: {input_path}")
        output_path = _resolve(args.output, paths.root) or paths.default_output_path(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.input_size <= 0 or args.input_size % 14:
            parser.error("--input-size must be a positive multiple of 14")
        if args.max_res <= 0 and args.max_res != -1:
            parser.error("--max-res must be a positive resolution or -1 to keep the source size")

        model = build_depth_model(
            encoder=encoder,
            checkpoint=_resolve(args.checkpoint, paths.root),
            project_root=paths.root,
            fp32=args.fp32,
            streaming=args.streaming,
            metric=metric,
            device=args.device,
        )
        run_video_inference(
            model=model,
            input_path=input_path,
            output_path=output_path,
            input_size=args.input_size,
            fp32=args.fp32,
            streaming=args.streaming,
            device=args.device,
            max_len=args.max_len,
            target_fps=args.target_fps,
            max_res=args.max_res,
            depth_map=_resolve(args.depth_map, paths.root),
            depth_dtype=args.depth_dtype,
            keep_depth_map=not args.discard_depth_map,
            video_crf=args.video_crf,
            audio=args.audio == "true",
            colormap=args.colormap,
        )
    except (ModelNotConfiguredError, RuntimeError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0
