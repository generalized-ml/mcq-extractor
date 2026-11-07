"""Lightweight text detection/recognition on existing question crops; no layout model."""
from pathlib import Path
import math
import re
import time


class MobileReader:
    def __init__(self, model=None, *, device="auto", local_files_only=False, **kwargs):
        if device == "mps":
            raise ValueError("Paddle mobile uses PaddlePaddle, not PyTorch MPS; use --device cpu")
        if local_files_only:
            raise ValueError("--local-files-only is supported only by --ocr-backend vl")
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("Install mobile dependencies: python -m pip install -e '.[paddle-mobile]'") from exc
        self.device = "gpu:0" if device == "cuda" else "cpu"
        self.model_id = "PP-OCRv5_mobile_det+PP-OCRv5_mobile_rec"
        self.revision = "PP-OCRv5"
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=0.0,
            device=self.device,
        )

    def read(self, path: Path) -> dict:
        start = time.monotonic()
        lines = []
        for result in self.ocr.predict(str(path)):
            texts, scores = result["rec_texts"], result["rec_scores"]
            if len(texts) != len(scores):
                raise ValueError("PaddleOCR returned mismatched texts and confidence scores")
            for text, score in zip(texts, scores):
                lines.append({"text": str(text), "score": float(score)})
        confidence = line_confidence(lines)
        return {"text": "\n".join(line["text"] for line in lines), "lines": lines,
                "confidence": confidence, "confidence_kind": "uncalibrated_recognition_line_score",
                "truncated": False, "seconds": time.monotonic() - start,
                "model": self.model_id, "revision": self.revision, "device": self.device}


def line_confidence(lines):
    scores = [line["score"] for line in lines if line["text"].strip()]
    valid = bool(scores) and all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
    return {
        "score": (math.exp(sum(math.log(max(s, 1e-12)) for s in scores) / len(scores)) if valid else None),
        "low_token_fraction": sum(s < .5 for s in scores) / len(scores) if valid else None,
        "scored_lines": len(scores),
    }


def scope_confidence(parts):
    """Scope mobile confidence across crop continuations without trimming OCR text.

    Paddle returns line scores, so the boundary line contributes its original
    score even if the cutoff falls within it. No word confidence is invented.
    """
    text = "\n\n".join(part["text"] for part in parts)
    marker = re.search(r"\b(?:solution|answer)\b", text, re.IGNORECASE)
    end = len(text)
    if marker:
        # Use original offsets; punctuation-only tokens do not count as words.
        words = list(re.finditer(r"[^\s]+", text[marker.end():]))
        words = [word for word in words if re.search(r"\w", word.group())]
        end = marker.end() + words[min(2, len(words) - 1)].end() if words else marker.end()
    offset = 0
    selected = []
    for part in parts:
        local_end = max(0, min(len(part["text"]), end - offset))
        included, position = [], 0
        for line in part["lines"]:
            if position < local_end and line["text"].strip():
                included.append(line)
            position += len(line["text"]) + 1
        part["confidence"] = line_confidence(included)
        part["confidence_scope"] = {
            "rule": "through_first_solution_or_answer_plus_3_words",
            "marker_found": bool(marker), "text": part["text"][:local_end],
            "excluded": offset >= end and bool(marker),
            "granularity": "line_score_including_boundary_line",
        }
        if not part["confidence_scope"]["excluded"]:
            selected.append(part)
        offset += len(part["text"]) + 2
    return selected
