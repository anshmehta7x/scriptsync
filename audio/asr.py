from __future__ import annotations

from typing import List

from .types import SpeechSegment


class FastWhisperASR:
    def __init__(self, model_size: str = "large-v3", device: str = "auto", compute_type: str = "auto"):
        from faster_whisper import WhisperModel

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self._whisper_cls = WhisperModel
        self.model, self.active_model_size, self.active_device, self.active_compute_type = self._init_model_with_fallback(
            WhisperModel=WhisperModel,
            model_size=model_size,
            preferred_device=device,
            compute_type=compute_type,
        )

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda failed" in msg

    @staticmethod
    def _init_model_with_fallback(WhisperModel, model_size: str, preferred_device: str, compute_type: str):
        candidates = []
        seen = set()

        def add_candidate(ms: str, dev: str, ctype: str):
            key = (ms, dev, ctype)
            if key not in seen:
                seen.add(key)
                candidates.append(key)

        if preferred_device == "cuda":
            if compute_type == "auto":
                add_candidate(model_size, "cuda", "float16")
                add_candidate(model_size, "cuda", "int8_float16")
                add_candidate(model_size, "cuda", "int8")
            else:
                add_candidate(model_size, "cuda", compute_type)

            if model_size.startswith("large"):
                add_candidate("medium", "cuda", "int8_float16")
                add_candidate("medium", "cuda", "int8")
                add_candidate("small", "cuda", "int8")

            add_candidate(model_size, "cpu", "int8")
            if model_size.startswith("large"):
                add_candidate("medium", "cpu", "int8")
                add_candidate("small", "cpu", "int8")
        else:
            ctype = "int8" if compute_type == "auto" else compute_type
            add_candidate(model_size, preferred_device, ctype)

        last_exc = None
        for ms, dev, ctype in candidates:
            try:
                model = WhisperModel(ms, device=dev, compute_type=ctype)
                print(f"ASR init: model={ms} device={dev} compute_type={ctype}")
                return model, ms, dev, ctype
            except RuntimeError as exc:
                last_exc = exc
                is_oom = FastWhisperASR._is_oom_error(exc)
                if is_oom:
                    print(f"⚠️ ASR init failed ({ms}, {dev}, {ctype}) due to OOM/CUDA issue; trying fallback...")
                    continue
                raise

        raise RuntimeError(
            "Failed to initialize faster-whisper model after fallbacks. "
            "Try --device cpu or --asr-model medium/small."
        ) from last_exc

    def _reload_model(self, model_size: str, device: str, compute_type: str) -> None:
        self.model = self._whisper_cls(model_size, device=device, compute_type=compute_type)
        self.active_model_size = model_size
        self.active_device = device
        self.active_compute_type = compute_type
        print(f"ASR reload: model={model_size} device={device} compute_type={compute_type}")

    def _run_transcribe(self, audio_path: str, source_lang: str, beam_size: int, best_of: int):
        return self.model.transcribe(
            audio_path,
            language=source_lang,
            vad_filter=True,
            beam_size=beam_size,
            best_of=best_of,
            condition_on_previous_text=True,
            vad_parameters={"min_silence_duration_ms": 250},
        )

    def transcribe(self, audio_path: str, source_lang: str) -> List[SpeechSegment]:
        try:
            segments, _ = self._run_transcribe(audio_path, source_lang, beam_size=5, best_of=5)
            segments = list(segments)
        except RuntimeError as exc:
            if not self._is_oom_error(exc):
                raise

            print("⚠️ ASR transcribe OOM with beam=5/best_of=5; retrying with lighter decode...")
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                segments, _ = self._run_transcribe(audio_path, source_lang, beam_size=1, best_of=1)
                segments = list(segments)
            except RuntimeError as exc2:
                if not self._is_oom_error(exc2):
                    raise

                print("⚠️ ASR still OOM; stepping down ASR model/device...")

                fallback_models = [
                    ("medium", "cuda", "int8_float16"),
                    ("medium", "cuda", "int8"),
                    ("small", "cuda", "int8"),
                    ("medium", "cpu", "int8"),
                    ("small", "cpu", "int8"),
                ]

                last_exc = exc2
                for ms, dev, ctype in fallback_models:
                    if (ms, dev, ctype) == (self.active_model_size, self.active_device, self.active_compute_type):
                        continue
                    try:
                        self._reload_model(ms, dev, ctype)
                        segments, _ = self._run_transcribe(audio_path, source_lang, beam_size=1, best_of=1)
                        segments = list(segments)
                        break
                    except RuntimeError as exc3:
                        last_exc = exc3
                        if self._is_oom_error(exc3):
                            try:
                                import torch

                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            continue
                        raise
                else:
                    raise RuntimeError(
                        "ASR transcribe failed after OOM fallbacks. "
                        "Try --asr-model medium/small or --device cpu."
                    ) from last_exc

        out: List[SpeechSegment] = []
        for seg in segments:
            txt = (seg.text or "").strip()
            if txt:
                out.append(SpeechSegment(start=float(seg.start), end=float(seg.end), text=txt))
        return out
