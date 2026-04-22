# scriptsync

Local pipeline for translating both:
- **On-screen text in video frames** (OCR + redraw)
- **Spoken audio track** (ASR + text translation + TTS dubbing)

## End-to-end local AV translation

Run:

```bash
python video/av_translate.py \
	--input input.mp4 \
	--output translated_output.mp4 \
	--source-lang fr \
	--target-lang hi \
	--device auto \
	--preset balanced
```

## Presets (speed vs quality)

### Turbo (max speed)

```bash
python video/av_translate.py \
	--input input.mp4 \
	--output translated_turbo.mp4 \
	--source-lang fr \
	--target-lang hi \
	--device auto \
	--preset turbo
```

### Balanced (default)

```bash
python video/av_translate.py \
	--input input.mp4 \
	--output translated_balanced.mp4 \
	--source-lang fr \
	--target-lang hi \
	--device auto \
	--preset balanced
```

### Quality (max quality)

```bash
python video/av_translate.py \
	--input input.mp4 \
	--output translated_quality.mp4 \
	--source-lang fr \
	--target-lang hi \
	--device auto \
	--preset quality
```

You can still override any preset parameter manually. Example:

```bash
python video/av_translate.py \
	--input input.mp4 \
	--output translated_custom.mp4 \
	--source-lang fr \
	--target-lang hi \
	--preset turbo \
	--asr-model large-v3 \
	--difference-threshold 0.06
```

## What it does

1. Translates on-screen text using the fast multithreaded frame pipeline.
2. Extracts source audio from the input video.
3. Runs local ASR via `faster-whisper`.
4. Translates transcript via local NLLB (`facebook/nllb-200-distilled-600M`).
5. Synthesizes local speech via MMS-TTS (`facebook/mms-tts-*`).
6. Muxes translated video + dubbed audio into final MP4.

## Speed/quality notes

- Use GPU for best speed (`--device cuda` or `--device auto` with CUDA available).
- Presets:
	- `turbo`: fastest practical settings
	- `balanced`: default tradeoff
	- `quality`: higher-quality ASR/translation and denser frame processing
- Frame translation speed is controlled by:
	- `--difference-threshold`
	- `--max-consecutive-skips`
	- `--max-workers`
	- `--gpu-concurrency-limit`

## Requirements

- `ffmpeg` and `ffprobe` installed and available in PATH.
- Python dependencies in `requirements.txt`.
- First run downloads model weights locally (Whisper, NLLB, MMS-TTS).
