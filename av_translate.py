from __future__ import annotations

from audio.asr import FastWhisperASR
from audio.languages import ISO6391_TO_ISO6393, NLLB_LANG_MAP
from audio.media import (
    build_dubbed_audio,
    extract_audio_wav,
    fit_audio_to_duration,
    get_duration_seconds,
    mux_video_and_audio,
)
from audio.translation import NLLBTranslator
from audio.tts import MMSTTS
from audio.types import SpeechSegment
from cli import build_parser, main
from presets import PRESET_CONFIG, apply_preset
from runner import run_pipeline

__all__ = [
    "FastWhisperASR",
    "ISO6391_TO_ISO6393",
    "MMSTTS",
    "NLLBTranslator",
    "NLLB_LANG_MAP",
    "PRESET_CONFIG",
    "SpeechSegment",
    "apply_preset",
    "build_dubbed_audio",
    "build_parser",
    "extract_audio_wav",
    "fit_audio_to_duration",
    "get_duration_seconds",
    "main",
    "mux_video_and_audio",
    "run_pipeline",
]

if __name__ == "__main__":
    main()
