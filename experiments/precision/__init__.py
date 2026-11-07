"""Convert the native OCR checkpoints to local float16 models."""

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
DET_FP16 = MODEL_DIR / "PP-OCRv5_mobile_det_safetensors_fp16"
REC_FP16 = MODEL_DIR / "PP-OCRv5_mobile_rec_safetensors_fp16"
