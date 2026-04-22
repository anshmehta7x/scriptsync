# scriptsync

Local pipeline for translating both:

- **On-screen text in video frames** using OCR + redraw
- **Spoken audio tracks** using ASR + translation + TTS dubbing

## Project layout

- `av_translate.py` — root entrypoint for the end-to-end AV pipeline
- `cli.py` — root CLI wiring
- `runner.py` — root orchestration for the AV flow
- `presets.py` — root preset configuration
- `audio/` — audio-specific modules (`asr`, `tts`, `media`, `translation`, `types`)
- `video/` — video-specific modules (`ocr`, frame rendering, video pipeline)

## End-to-end local AV translation

Run:

```bash
python av_translate.py \
  --input input.mp4 \
  --output translated_output.mp4 \
  --source-lang fr \
  --target-lang hi \
  --device auto \
  --preset balanced
```

## Presets

### Turbo

```bash
python av_translate.py \
  --input input.mp4 \
  --output translated_turbo.mp4 \
  --source-lang fr \
  --target-lang hi \
  --device auto \
  --preset turbo
```

### Balanced

```bash
python av_translate.py \
  --input input.mp4 \
  --output translated_balanced.mp4 \
  --source-lang fr \
  --target-lang hi \
  --device auto \
  --preset balanced
```

### Quality

```bash
python av_translate.py \
  --input input.mp4 \
  --output translated_quality.mp4 \
  --source-lang fr \
  --target-lang hi \
  --device auto \
  --preset quality
```

You can override preset parameters manually. Example:

```bash
python av_translate.py \
  --input input.mp4 \
  --output translated_custom.mp4 \
  --source-lang fr \
  --target-lang hi \
  --preset turbo \
  --asr-model large-v3 \
  --difference-threshold 0.06
```

## What the pipeline does

1. Translates on-screen text with the multithreaded video pipeline.
2. Extracts source audio from the input video.
3. Runs ASR with `faster-whisper`.
4. Translates transcript text with NLLB (`facebook/nllb-200-distilled-600M`).
5. Synthesizes dubbed speech with MMS-TTS (`facebook/mms-tts-*`).
6. Muxes the translated video and dubbed audio into the final MP4.

## Key options

- `--input` — input video path
- `--output` — final translated output path
- `--source-lang` — source language ISO-639-1 code
- `--target-lang` — target language ISO-639-1 code
- `--preset` — one of `turbo`, `balanced`, or `quality`
- `--device` — `auto`, `cuda`, or `cpu`

Advanced frame-processing controls:

- `--difference-threshold`
- `--max-consecutive-skips`
- `--max-workers`
- `--gpu-concurrency-limit`
- `--queue-size`
- `--frame-timeout`

## Speed and quality notes

- Use GPU for best speed with `--device cuda`, or `--device auto` if CUDA is available.
- `turbo` favors speed.
- `balanced` is the default tradeoff.
- `quality` uses denser frame processing and more conservative settings.

## Requirements

- `ffmpeg` and `ffprobe` installed and available in `PATH`
- Python dependencies from `requirements.txt`
- Internet access on first run to download model weights

## Install

```bash
pip install -r requirements.txt
```

## Example

```bash
python av_translate.py \
  --input demo/input.mp4 \
  --output demo/output_hi.mp4 \
  --source-lang fr \
  --target-lang hi \
  --device auto \
  --preset balanced
```
