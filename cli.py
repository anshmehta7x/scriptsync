from __future__ import annotations

import argparse

from presets import PRESET_CONFIG, apply_preset
from runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local end-to-end video + audio translation pipeline"
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument(
        "--output", required=True, help="Final translated output video path"
    )
    parser.add_argument(
        "--source-lang",
        default="en",
        help="Source language ISO-639-1 code",
    )
    parser.add_argument(
        "--target-lang",
        default="fr",
        help="Target language ISO-639-1 code",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIG.keys()),
        default="balanced",
        help="Tuned speed/quality preset",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Model execution device",
    )

    parser.add_argument(
        "--asr-model",
        default=None,
        help="faster-whisper model size (overrides preset)",
    )
    parser.add_argument(
        "--translation-model",
        default=None,
        help="HuggingFace NLLB model id (overrides preset)",
    )
    parser.add_argument(
        "--translation-batch-size",
        type=int,
        default=None,
        help="Batch size for translation generation (overrides preset)",
    )

    parser.add_argument(
        "--difference-threshold",
        type=float,
        default=None,
        help="Video frame diff threshold (overrides preset)",
    )
    parser.add_argument(
        "--max-consecutive-skips",
        type=int,
        default=None,
        help="Force frame refresh interval (overrides preset)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Video OCR worker threads",
    )
    parser.add_argument(
        "--gpu-concurrency-limit",
        type=int,
        default=None,
        help="Concurrent GPU OCR jobs (overrides preset)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=None,
        help="Frame queue size (overrides preset)",
    )
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=None,
        help="Max seconds to wait for next frame before forced skip (overrides preset)",
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temp intermediate files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args = apply_preset(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()
