from __future__ import annotations

import numpy as np
from typing import Any

from .languages import ISO6391_TO_ISO6393


class MMSTTS:
    def __init__(self, target_lang: str, device: str = "auto"):
        import torch
        from transformers import AutoTokenizer, VitsModel

        iso3 = ISO6391_TO_ISO6393.get(target_lang)
        if not iso3:
            raise ValueError(
                f"Unsupported MMS TTS language mapping for '{target_lang}'"
            )

        model_id = f"facebook/mms-tts-{iso3}"

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model: Any = VitsModel.from_pretrained(model_id)
        model = model.to(self.device)
        model.eval()
        self.model = model
        self.sample_rate = int(self.model.config.sampling_rate)

    def synthesize(self, text: str) -> np.ndarray:
        import torch

        text = (text or "").strip()
        if not text:
            return np.zeros(1, dtype=np.float32)

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            waveform = self.model(**inputs).waveform

        audio = waveform.squeeze().detach().float().cpu().numpy()
        if audio.ndim == 0:
            audio = np.array([float(audio)], dtype=np.float32)
        return audio.astype(np.float32)
