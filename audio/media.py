from __future__ import annotations

import subprocess
from typing import Sequence

import numpy as np
from scipy.io import wavfile

from .tts import MMSTTS
from .types import SpeechSegment


def _run(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)


def extract_audio_wav(video_path: str, wav_path: str, sample_rate: int = 16000) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            wav_path,
        ]
    )


def get_duration_seconds(media_path: str) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            media_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float((proc.stdout or "0").strip() or 0.0)


def fit_audio_to_duration(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 0:
        return np.zeros(1, dtype=np.float32)

    if audio.size <= 1:
        return np.zeros(target_samples, dtype=np.float32)

    if audio.size == target_samples:
        return audio.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, num=audio.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_samples, dtype=np.float32)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def build_dubbed_audio(
    segments: list[SpeechSegment],
    translated_texts: list[str],
    tts: MMSTTS,
    output_wav_path: str,
    total_duration_sec: float,
) -> None:
    sr = tts.sample_rate
    total_samples = max(int(total_duration_sec * sr), 1)
    track = np.zeros(total_samples, dtype=np.float32)

    for seg, txt in zip(segments, translated_texts):
        start_idx = int(seg.start * sr)
        if start_idx >= total_samples:
            continue

        segment_duration = max(seg.end - seg.start, 0.05)
        target_len = max(1, int(segment_duration * sr))

        spoken = tts.synthesize(txt)
        spoken = fit_audio_to_duration(spoken, target_len)

        end_idx = min(start_idx + spoken.size, total_samples)
        usable = end_idx - start_idx
        if usable > 0:
            track[start_idx:end_idx] += spoken[:usable] * 0.9

    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0:
        track = track / max(peak / 0.95, 1.0)

    pcm16 = np.clip(track * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(output_wav_path, sr, pcm16)


def mux_video_and_audio(
    video_no_audio: str, dubbed_audio_wav: str, output_video: str
) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_no_audio,
            "-i",
            dubbed_audio_wav,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            output_video,
        ]
    )
