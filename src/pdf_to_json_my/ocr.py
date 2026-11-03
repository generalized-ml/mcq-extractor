"""Native PyTorch PP-OCRv5 inference; no PaddleOCR or PaddlePaddle dependency."""
from pathlib import Path
import time

from .confidence import line_confidence

DET_MODEL = "PaddlePaddle/PP-OCRv5_mobile_det_safetensors"
REC_MODEL = "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors"
DET_REVISION = "c5041d225cf951ff06900ab81a3c7d543c45e2ad"
REC_REVISION = "485a9a9781535ad477de3b1eea5f96a2e7300d8d"


def cached_model(model, revision, *, local_files_only=False):
    """Use local directory/cache first; fetch only missing model assets from HF."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path.resolve())
    options = dict(repo_id=model, revision=revision,
                   allow_patterns=["*.json", "*.safetensors"])
    try:
        snapshot = snapshot_download(**options, local_files_only=True)
        # A snapshot can exist after a partial/interrupted download.
        if (Path(snapshot) / 'model.safetensors').is_file() and all(
            (Path(snapshot) / name).is_file() for name in ('config.json', 'preprocessor_config.json')
        ):
            return snapshot
    except LocalEntryNotFoundError:
        pass
    if local_files_only:
        raise RuntimeError(f"Model not fully cached: {model}@{revision}")
    return snapshot_download(**options)


def reading_order(boxes):
    """Top-to-bottom, with near-aligned text boxes read left-to-right."""
    ordered = sorted(boxes, key=lambda b: (min(p[1] for p in b), min(p[0] for p in b)))
    for i in range(1, len(ordered)):
        j = i
        while j > 0:
            a, b = ordered[j-1], ordered[j]
            ay, by = min(p[1] for p in a), min(p[1] for p in b)
            tolerance = .5 * min(max(p[1] for p in a)-ay, max(p[1] for p in b)-by)
            if abs(ay-by) <= tolerance and min(p[0] for p in b) < min(p[0] for p in a):
                ordered[j-1], ordered[j] = b, a
                j -= 1
            else:
                break
    return ordered


def rectify(image, box):
    """Perspective crop from detector TL, TR, BR, BL polygon coordinates."""
    import cv2
    import numpy as np
    from PIL import Image
    points = np.asarray(box, dtype=np.float32)
    width = max(1, round(max(np.linalg.norm(points[0]-points[1]), np.linalg.norm(points[2]-points[3]))))
    height = max(1, round(max(np.linalg.norm(points[0]-points[3]), np.linalg.norm(points[1]-points[2]))))
    target = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    return Image.fromarray(cv2.warpPerspective(image, matrix, (width, height),
                                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))


class HFMobileReader:
    def __init__(self, *, det_model=DET_MODEL, rec_model=REC_MODEL,
                 det_revision=DET_REVISION, rec_revision=REC_REVISION,
                 device="cpu", batch_size=8, local_files_only=False):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoModelForTextRecognition
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; select --device cpu")
        self.device, self.batch_size = device, batch_size
        det = cached_model(det_model, det_revision, local_files_only=local_files_only)
        rec = cached_model(rec_model, rec_revision, local_files_only=local_files_only)
        self.det_processor = AutoImageProcessor.from_pretrained(det, local_files_only=True)
        self.rec_processor = AutoImageProcessor.from_pretrained(rec, local_files_only=True)
        self.detector = AutoModelForObjectDetection.from_pretrained(det, local_files_only=True).to(device).eval()
        self.recognizer = AutoModelForTextRecognition.from_pretrained(rec, local_files_only=True).to(device).eval()
        self.model_info = {"detector": det, "recognizer": rec}

    def read(self, path):
        import torch
        import numpy as np
        from PIL import Image
        start = time.monotonic()
        with Image.open(path) as source:
            image = source.convert('RGB')
        lines = []
        try:
            with torch.inference_mode():
                inputs = self.det_processor(images=image, return_tensors='pt').to(self.device)
                target_sizes = inputs.pop('target_sizes')
                detection = self.det_processor.post_process_object_detection(
                    self.detector(**inputs), target_sizes=target_sizes)[0]
                boxes = reading_order(detection['boxes'].cpu().tolist())
                pixels = np.asarray(image)
                for start_index in range(0, len(boxes), self.batch_size):
                    batch_boxes = boxes[start_index:start_index+self.batch_size]
                    crops = [rectify(pixels, box) for box in batch_boxes]
                    try:
                        # Process independently to preserve each line's aspect ratio.
                        # The upstream batched processor stretches every image to
                        # one width. Pad normalized tensors instead.
                        tensors = [self.rec_processor(images=crop, return_tensors='pt')['pixel_values']
                                   for crop in crops]
                        width = max(t.shape[-1] for t in tensors)
                        batch = {'pixel_values': torch.cat([
                            torch.nn.functional.pad(t, (0, width-t.shape[-1])) for t in tensors
                        ]).to(self.device)}
                        recognized = self.rec_processor.post_process_text_recognition(self.recognizer(**batch))
                    finally:
                        for crop in crops:
                            crop.close()
                    if len(recognized) != len(batch_boxes):
                        raise RuntimeError('Recognizer returned an unexpected number of lines')
                    for box, result in zip(batch_boxes, recognized):
                        score = float(result['score'])
                        # CTC produces NaN for an all-blank prediction; keep it as zero confidence.
                        if not np.isfinite(score):
                            score = 0.0
                        lines.append({'text': result['text'], 'score': score, 'box': box})
        finally:
            image.close()
        return {'text': '\n'.join(line['text'] for line in lines), 'lines': lines,
                'confidence': line_confidence(lines), 'confidence_kind': 'uncalibrated_recognition_line_score',
                'truncated': False, 'seconds': time.monotonic()-start, 'device': self.device,
                'models': self.model_info}
