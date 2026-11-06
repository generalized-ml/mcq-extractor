"""Compare FP32 serial OCR with persistent CPU recognition workers on one crop."""

import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .ocr import HFMobileReader, REC_MODEL, REC_REVISION, cached_model, reading_order, recognize_crops, rectify
from .profiling import OCRProfiler

_worker = None


def initialize_worker(model_path, threads, ready):
    import cv2
    import torch
    from transformers import AutoImageProcessor, AutoModelForTextRecognition
    global _worker
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForTextRecognition.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32).eval()
    _worker = (processor, model)
    ready.put(os.getpid())


def worker_identity():
    return os.getpid()


def recognize_batch(payload):
    import torch
    from PIL import Image
    processor, model = _worker
    crops = [Image.fromarray(pixels) for pixels in payload]
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            recognized = recognize_crops(crops, processor, model, 'cpu', OCRProfiler(None))
        return {'recognized': recognized, 'seconds': time.perf_counter() - started,
                'pid': os.getpid()}
    finally:
        for crop in crops:
            crop.close()


def parallel_read(reader, path, executor):
    import numpy as np
    import torch
    from PIL import Image
    start = time.perf_counter()
    with Image.open(path) as source:
        image = source.convert('RGB')
    try:
        with torch.inference_mode():
            inputs = reader.det_processor(images=image, return_tensors='pt').to('cpu')
            target_sizes = inputs.pop('target_sizes')
            inputs['pixel_values'] = inputs['pixel_values'].to(reader.detector.dtype)
            output = reader.detector(**inputs)
            output.last_hidden_state = output.last_hidden_state.float()
            detection = reader.det_processor.post_process_object_detection(output, target_sizes=target_sizes)[0]
            boxes = reading_order(detection['boxes'].cpu().tolist())
        detection_seconds = time.perf_counter() - start
        crop_start = time.perf_counter()
        pixels = np.asarray(image)
        batches = []
        for index in range(0, len(boxes), reader.batch_size):
            batch = []
            for box in boxes[index:index + reader.batch_size]:
                crop = rectify(pixels, box)
                try:
                    batch.append(np.array(crop))
                finally:
                    crop.close()
            batches.append(batch)
        cropping_seconds = time.perf_counter() - crop_start
        recognition_start = time.perf_counter()
        # map preserves submission order even when batches finish out of order.
        results = list(executor.map(recognize_batch, batches, chunksize=1))
        recognition_seconds = time.perf_counter() - recognition_start
        recognized = [line for batch in results for line in batch['recognized']]
        if len(recognized) != len(boxes):
            raise RuntimeError('Unexpected number of recognition results')
        lines = []
        for box, result in zip(boxes, recognized):
            score = float(result['score'])
            lines.append({'box': box, 'text': result['text'],
                          'score': score if np.isfinite(score) else 0.0})
        return {'text': '\n'.join(line['text'] for line in lines), 'lines': lines,
                'seconds': time.perf_counter() - start,
                'detection_seconds': detection_seconds, 'cropping_seconds': cropping_seconds,
                'recognition_wall_seconds': recognition_seconds, 'worker_batches': results}
    finally:
        image.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('crop', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads-per-worker', type=int, default=2)
    parser.add_argument('--serial-threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--runs', type=int, default=3, help='First pass plus warm repeats (minimum 2)')
    parser.add_argument('--output', type=Path, default=Path('outputs/ocr_timing/multiprocessing.json'))
    args = parser.parse_args()
    if min(args.workers, args.threads_per_worker, args.serial_threads, args.batch_size) < 1 or args.runs < 2:
        parser.error('Worker/thread/batch counts must be positive; --runs must be at least 2')
    import torch
    import cv2
    torch.set_num_threads(args.serial_threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    report = {'crop': str(args.crop), 'precision': 'float32', 'device': 'cpu',
              'settings': vars(args).copy(), 'serial': [], 'parallel': []}
    report['settings'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['settings'].items()}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    started = time.perf_counter()
    reader = HFMobileReader(device='cpu', batch_size=args.batch_size, local_files_only=True)
    assert str(reader.recognizer.dtype) == str(reader.detector.dtype) == 'torch.float32'
    report['serial_initialization_seconds'] = time.perf_counter() - started
    print(f"Serial initialization: {report['serial_initialization_seconds']:.3f}s", flush=True)
    for index in range(args.runs):
        result = reader.read(args.crop)
        report['serial'].append(result)
        print(f"Serial pass {index + 1}: {result['seconds']:.3f}s; {len(result['lines'])} lines", flush=True)
        save()

    # Parent retains only detection models during the parallel measurement.
    del reader.recognizer, reader.rec_processor
    import gc
    gc.collect()
    torch.set_num_threads(args.threads_per_worker)
    model_path = cached_model(REC_MODEL, REC_REVISION, local_files_only=True)
    context = mp.get_context('spawn')
    ready = context.Queue()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                             initializer=initialize_worker,
                             initargs=(model_path, args.threads_per_worker, ready)) as executor:
        startup = [executor.submit(worker_identity) for _ in range(args.workers)]
        for future in startup:
            future.result()
        report['worker_pids'] = [ready.get(timeout=60) for _ in range(args.workers)]
        report['worker_initialization_seconds'] = time.perf_counter() - started
        print(f"Workers ready: {report['worker_pids']}; startup {report['worker_initialization_seconds']:.3f}s", flush=True)
        for index in range(args.runs):
            result = parallel_read(reader, args.crop, executor)
            result['text_matches_serial'] = result['text'] == report['serial'][0]['text']
            result['boxes_match_serial'] = [line['box'] for line in result['lines']] == [
                line['box'] for line in report['serial'][0]['lines']]
            report['parallel'].append(result)
            print(f"Parallel pass {index + 1}: {result['seconds']:.3f}s; "
                  f"recognition {result['recognition_wall_seconds']:.3f}s; "
                  f"text matches: {result['text_matches_serial']}", flush=True)
            save()
    ready.close()
    ready.join_thread()
    serial = statistics.median(result['seconds'] for result in report['serial'][1:])
    parallel = statistics.median(result['seconds'] for result in report['parallel'][1:])
    report['summary'] = {'serial_warm_median_seconds': serial, 'parallel_warm_median_seconds': parallel,
                         'speedup': serial / parallel,
                         'all_parallel_texts_match': all(r['text_matches_serial'] for r in report['parallel']),
                         'all_parallel_boxes_match': all(r['boxes_match_serial'] for r in report['parallel'])}
    save()
    print(json.dumps(report['summary'], indent=2), flush=True)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
