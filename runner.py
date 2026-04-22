from __future__ import annotations

import argparse
import gc
import os
import shutil
import tempfile
from typing import Dict

from audio.asr import FastWhisperASR
from audio.media import (
    build_dubbed_audio,
    extract_audio_wav,
    get_duration_seconds,
    mux_video_and_audio,
)
from audio.translation import NLLBTranslator
from audio.tts import MMSTTS
from video.main import FastPipeline


def _clear_torch_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_pipeline(args: argparse.Namespace) -> None:
    print(
        f"Preset={args.preset} | ASR={args.asr_model} | "
        f"NLLB={args.translation_model} | batch={args.translation_batch_size}"
    )
    print(
        f"Video knobs: diff={args.difference_threshold} | skips={args.max_consecutive_skips} | "
        f"workers={args.max_workers} | gpu_limit={args.gpu_concurrency_limit} | queue={args.queue_size} | "
        f"frame_timeout={args.frame_timeout}"
    )

    work_dir = tempfile.mkdtemp(prefix="scriptsync_av_")
    try:
        video_no_audio = os.path.join(work_dir, "video_translated.mp4")
        source_wav = os.path.join(work_dir, "source_audio.wav")
        dubbed_wav = os.path.join(work_dir, "dubbed_audio.wav")
        shared_translation_cache: Dict[str, str] = {}

        print("\n[1/5] Translating on-screen video text...")
        FastPipeline(
            video_path=args.input,
            output_path=video_no_audio,
            source_lang=args.source_lang,
            dest_lang=args.target_lang,
            difference_threshold=args.difference_threshold,
            max_consecutive_skips=args.max_consecutive_skips,
            max_workers=args.max_workers,
            gpu_concurrency_limit=args.gpu_concurrency_limit,
            queue_size=args.queue_size,
            frame_timeout=args.frame_timeout,
            encode=True,
            compress_output=False,
            shared_translation_cache=shared_translation_cache,
        ).run()

        print("\n[2/5] Extracting source audio...")
        extract_audio_wav(args.input, source_wav, sample_rate=16000)

        print("\n[3/5] Running ASR + translation...")
        asr = FastWhisperASR(model_size=args.asr_model, device=args.device)
        segments = asr.transcribe(source_wav, source_lang=args.source_lang)
        if not segments:
            raise RuntimeError("No speech segments detected in source audio.")

        del asr
        gc.collect()
        _clear_torch_cuda_cache()

        translator = NLLBTranslator(model_id=args.translation_model, device=args.device)
        raw_texts = [segment.text for segment in segments]

        misses = [text for text in raw_texts if text not in shared_translation_cache]
        if misses:
            translated_misses = translator.translate_batch(
                misses,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                batch_size=args.translation_batch_size,
            )
            for source_text, translated_text in zip(misses, translated_misses):
                shared_translation_cache[source_text] = translated_text

        translated_texts = [shared_translation_cache[text] for text in raw_texts]

        print("\n[4/5] Synthesizing dubbed audio...")
        tts = MMSTTS(target_lang=args.target_lang, device=args.device)
        total_duration = get_duration_seconds(args.input)
        build_dubbed_audio(
            segments=segments,
            translated_texts=translated_texts,
            tts=tts,
            output_wav_path=dubbed_wav,
            total_duration_sec=total_duration,
        )

        print("\n[5/5] Muxing translated video + dubbed audio...")
        mux_video_and_audio(video_no_audio, dubbed_wav, args.output)

        print("\n✅ Done")
        print(f"Output: {args.output}")
    finally:
        if getattr(args, "keep_temp", False):
            print(f"Temp files kept in: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
