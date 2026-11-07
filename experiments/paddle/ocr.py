"""PaddleOCR-VL-1.6 crop recognition through its Transformers implementation."""

from __future__ import annotations

import math
from pathlib import Path
import time

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
OCR_PROMPT = "OCR:"


def token_confidence(tokens: list[tuple[int, float]], special_ids: set[int]) -> dict:
    """Summarize generation likelihood, excluding control tokens; not calibrated."""
    scores = [score for token, score in tokens if token not in special_ids]
    if not scores or any(not math.isfinite(score) or score > 0 for score in scores):
        return {"score": None, "low_token_fraction": None, "scored_tokens": len(scores)}
    return {"score": math.exp(math.fsum(scores) / len(scores)),
            "low_token_fraction": sum(score < math.log(.5) for score in scores) / len(scores),
            "scored_tokens": len(scores)}


class PaddleReader:
    def __init__(self, model: str = MODEL_ID, *, revision: str = "main",
                 device: str = "auto", max_tokens: int = 2048,
                 local_files_only: bool = False):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install dependencies: python -m pip install -e '.[paddle]'") from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; use --device cpu")
        self.device, self.max_tokens = device, max_tokens
        self.model_id = model
        common = {"revision": revision, "local_files_only": local_files_only}
        self.processor = AutoProcessor.from_pretrained(model, **common)
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, dtype=dtype, attn_implementation="eager", **common).to(device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def read(self, path: Path) -> dict:
        import torch
        from PIL import Image
        from transformers import LogitsProcessor, LogitsProcessorList

        class CaptureProbability(LogitsProcessor):
            def __init__(self):
                self.values = []

            def __call__(self, input_ids, scores):
                # Greedy decoding: capture only the selected token's likelihood,
                # instead of retaining a vocabulary-sized score tensor per step.
                token = scores[0].argmax()
                log_probability = scores[0, token].float() - torch.logsumexp(scores[0].float(), dim=-1)
                self.values.append((token.detach(), log_probability.detach()))
                return scores

        with Image.open(path) as source:
            image = source.convert("RGB")
        try:
            messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                        {"type": "text", "text": OCR_PROMPT}]}]
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                                                        tokenize=True, return_dict=True,
                                                        return_tensors="pt").to(self.device)
        finally:
            image.close()
        capture = CaptureProbability()
        start = time.monotonic()
        with torch.inference_mode():
            sequences = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                            do_sample=False, num_beams=1,
                                            logits_processor=LogitsProcessorList([capture]))
        generated = sequences[0, inputs["input_ids"].shape[-1]:].tolist()
        tokens = [(int(token.item()), float(score.item())) for token, score in capture.values]
        if [token for token, _ in tokens] != generated:
            raise RuntimeError("Generated tokens differ from captured greedy scores")
        tokenizer = self.processor.tokenizer
        special = set(tokenizer.all_special_ids)
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        return {"text": self.processor.decode(generated, skip_special_tokens=True),
                "confidence": token_confidence(tokens, special),
                "confidence_kind": "uncalibrated_generation_token_probability",
                "truncated": len(generated) >= self.max_tokens and (not generated or generated[-1] not in eos_ids),
                "generated_tokens": len(generated), "seconds": time.monotonic() - start,
                "model": self.model_id, "revision": self.revision, "device": self.device,
                "prompt": OCR_PROMPT}
