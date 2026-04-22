import argparse
import os
from typing import TypedDict


class PresetConfig(TypedDict):
    asr_model: str
    translation_model: str
    translation_batch_size: int
    difference_threshold: float
    max_consecutive_skips: int
    max_workers: int | None
    gpu_concurrency_limit: int
    queue_size: int
    frame_timeout: float


PRESET_CONFIG: dict[str, PresetConfig] = {
    "turbo": {
        "asr_model": "medium",
        "translation_model": "facebook/nllb-200-distilled-600M",
        "translation_batch_size": 24,
        "difference_threshold": 0.11,
        "max_consecutive_skips": 900,
        "max_workers": max(2, (os.cpu_count() or 4) - 1),
        "gpu_concurrency_limit": 6,
        "queue_size": 1600,
        "frame_timeout": 6.0,
    },
    "balanced": {
        "asr_model": "medium",
        "translation_model": "facebook/nllb-200-distilled-600M",
        "translation_batch_size": 16,
        "difference_threshold": 0.045,
        "max_consecutive_skips": 180,
        "max_workers": None,
        "gpu_concurrency_limit": 4,
        "queue_size": 1200,
        "frame_timeout": 15.0,
    },
    "quality": {
        "asr_model": "medium",
        "translation_model": "facebook/nllb-200-distilled-600M",
        "translation_batch_size": 8,
        "difference_threshold": 0.03,
        "max_consecutive_skips": 90,
        "max_workers": max(2, (os.cpu_count() or 4) // 2),
        "gpu_concurrency_limit": 3,
        "queue_size": 900,
        "frame_timeout": 20.0,
    },
}


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset_cfg = PRESET_CONFIG[args.preset]

    if args.asr_model is None:
        args.asr_model = str(preset_cfg["asr_model"])
    if args.translation_model is None:
        args.translation_model = str(preset_cfg["translation_model"])
    if args.translation_batch_size is None:
        args.translation_batch_size = int(preset_cfg["translation_batch_size"])
    if args.difference_threshold is None:
        args.difference_threshold = float(preset_cfg["difference_threshold"])
    if args.max_consecutive_skips is None:
        args.max_consecutive_skips = int(preset_cfg["max_consecutive_skips"])
    if args.max_workers is None:
        args.max_workers = preset_cfg["max_workers"]
    if args.gpu_concurrency_limit is None:
        args.gpu_concurrency_limit = int(preset_cfg["gpu_concurrency_limit"])
    if args.queue_size is None:
        args.queue_size = int(preset_cfg["queue_size"])
    if args.frame_timeout is None:
        args.frame_timeout = float(preset_cfg["frame_timeout"])

    return args
