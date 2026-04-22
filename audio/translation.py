from __future__ import annotations

from typing import List

from .languages import NLLB_LANG_MAP


class NLLBTranslator:
    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", device: str = "auto"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.model = self._init_model_with_fallback(AutoModelForSeq2SeqLM, model_id, self.device)
        self.model.eval()

    @staticmethod
    def _init_model_with_fallback(AutoModelForSeq2SeqLM, model_id: str, preferred_device: str):
        import torch

        def load_model_for_init(**kwargs):
            try:
                return AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
            except ImportError:
                if kwargs.get("low_cpu_mem_usage"):
                    reduced = {k: v for k, v in kwargs.items() if k != "low_cpu_mem_usage"}
                    print("⚠️ 'accelerate' missing; loading NLLB without low_cpu_mem_usage")
                    return AutoModelForSeq2SeqLM.from_pretrained(model_id, **reduced)
                raise

        if preferred_device == "cuda":
            try:
                model = load_model_for_init(
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                model.to("cuda")
                print(f"NLLB init: model={model_id} device=cuda dtype=float16")
                return model
            except RuntimeError as exc:
                msg = str(exc).lower()
                is_oom = "out of memory" in msg or "cuda" in msg
                if not is_oom:
                    raise
                print("⚠️ NLLB CUDA init failed due to OOM/CUDA issue; falling back to CPU int8/float32 execution")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        model = load_model_for_init(low_cpu_mem_usage=True)
        model.to("cpu")
        print(f"NLLB init: model={model_id} device=cpu")
        return model

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str, batch_size: int = 16) -> List[str]:
        if not texts:
            return []

        src = NLLB_LANG_MAP.get(source_lang)
        dst = NLLB_LANG_MAP.get(target_lang)
        if not src or not dst:
            raise ValueError(f"Unsupported NLLB language mapping: {source_lang}->{target_lang}")

        import torch

        self.tokenizer.src_lang = src
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(dst)

        output: List[str] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=256,
                    num_beams=4,
                )

            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            output.extend(decoded)

        return output
