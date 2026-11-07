# Local float16 OCR models

From the repository root, convert the cached, pinned detector and recognizer:

```bash
.venv/bin/python -m experiments.precision --local-files-only
```

Omit `--local-files-only` to download missing original weights. The recognizer's
official repository name ends in `rec_safetensors` (plural).

Both converted model folders are saved under `model_quantization/models/`, beside
this code. The original Hugging Face cache stays intact. Each folder includes
the processor, config, FP16 safetensors weights, and a conversion report. Existing
destinations are refused; use `--output-dir` for another conversion.

Every floating-point checkpoint tensor is verified against its FP16 conversion;
integer counters retain their original dtype. This is a precision conversion,
not integer quantization. Weight storage is approximately halved; inference speed
and OCR accuracy must be measured on the target device.

To load these copies explicitly:

```python
from experiments.precision import DET_FP16, REC_FP16
from mcq_extractor.ocr import HFMobileReader

reader = HFMobileReader(det_model=str(DET_FP16), rec_model=str(REC_FP16),
                        device='cpu', batch_size=8, local_files_only=True)
```

The reader loads the checkpoint dtype and casts input pixels to match it. Its
default model locations still select the original models. The one-crop benchmark
at `outputs/ocr_timing/benchmark_native_one_crop.py` explicitly selects FP16 copies.
