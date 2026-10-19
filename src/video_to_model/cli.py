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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-to-model",
        description="Generate a depth video from an input video.",
    )
    parser.add_argument("input", type=Path, help="Input video path")
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
    parser.add_argument("--streaming", action="store_true", help="Use experimental streaming mode")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.model:
        metric = args.model.startswith("metric-")
        encoder = {"small": "vits", "base": "vitb", "large": "vitl"}[args.model.rsplit("-", 1)[1]]
        if args.metric or args.encoder != "vits":
            parser.error("Use either --model or the legacy --encoder/--metric options, not both")
    else:
        metric = args.metric
        encoder = args.encoder
    paths = ProjectPaths.from_project_root(Path.cwd())
    input_path = args.input if args.input.is_absolute() else paths.root / args.input
    output_path = args.output or paths.default_output_path(input_path)
    if not output_path.is_absolute():
        output_path = paths.root / output_path

    if not input_path.is_file():
        parser.error(f"Input video does not exist: {input_path}")
    if args.input_size <= 0 or args.input_size % 14:
        parser.error("--input-size must be a positive multiple of 14")

    try:
        model = build_depth_model(
            encoder=encoder,
            checkpoint=args.checkpoint,
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
        )
    except (ModelNotConfiguredError, RuntimeError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0
