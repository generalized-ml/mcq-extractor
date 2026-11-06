"""Optional wall-clock profiling for one OCR crop, including model submodules."""

import time
from collections import defaultdict
from contextlib import contextmanager


class OCRProfiler:
    def __init__(self, reader, enabled=False):
        self.reader = reader
        self.enabled = enabled
        self.seconds = defaultdict(float)
        self.handles = []

    def sync(self):
        # GPU operations are asynchronous; include completion in wall timings.
        import torch
        if self.reader.device.startswith('cuda'):
            torch.cuda.synchronize(self.reader.device)
        elif self.reader.device == 'mps':
            torch.mps.synchronize()

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.seconds[name] += time.perf_counter() - start

    def __enter__(self):
        if not self.enabled:
            return self
        modules = {
            'detector_backbone': self.reader.detector.model.backbone,
            'detector_feature_fusion': self.reader.detector.model.neck,
            'detector_head': self.reader.detector.head,
            'recognizer_backbone': self.reader.recognizer.model.backbone,
            'recognizer_svtr_encoder': self.reader.recognizer.head.encoder,
            'recognizer_classifier': self.reader.recognizer.head.head,
        }
        for name, module in modules.items():
            starts = []

            def before(module, args, starts=starts):
                self.sync()
                starts.append(time.perf_counter())

            def after(module, args, output, name=name, starts=starts):
                self.sync()
                self.seconds[name] += time.perf_counter() - starts.pop()

            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after))
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()

    def report(self, total):
        stages = dict(self.seconds)
        # Replace inclusive forward totals with unmeasured model work, so rows
        # can be added without double counting the hooked submodules.
        for prefix, components in (
            ('detector', ('backbone', 'feature_fusion', 'head')),
            ('recognizer', ('backbone', 'svtr_encoder', 'classifier')),
        ):
            inclusive = stages.pop(prefix + '_forward_total', 0.0)
            stages[prefix + '_other_model_ops'] = max(
                0.0, inclusive - sum(stages.get(prefix + '_' + name, 0.0) for name in components))
        stages['other_ocr_work'] = max(0.0, total - sum(stages.values()))
        return {'stages_seconds': stages, 'total_seconds': total,
                'note': 'Wall time with profiling overhead; recognition stages sum all line batches. '
                        'SVTR encoder includes its convolution layers and attention blocks.'}


def main():
    import argparse
    import json
    from pathlib import Path
    from .ocr import HFMobileReader

    parser = argparse.ArgumentParser(description='Profile each OCR stage on a saved crop.')
    parser.add_argument('crop', type=Path)
    parser.add_argument('--precision', choices=('float32', 'float16'), default='float32')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    options = {}
    if args.precision == 'float16':
        from .model_quantization import DET_FP16, REC_FP16
        options = {'det_model': str(DET_FP16), 'rec_model': str(REC_FP16)}
    started = time.perf_counter()
    reader = HFMobileReader(device=args.device, batch_size=args.batch_size,
                            local_files_only=True, **options)
    report = {'crop': str(args.crop), 'precision': args.precision, 'device': args.device,
              'batch_size': args.batch_size, 'models': reader.model_info,
              'detector_dtype': str(reader.detector.dtype),
              'recognizer_dtype': str(reader.recognizer.dtype),
              'initialization_seconds': time.perf_counter() - started, 'runs': []}
    print(f"Model initialization: {report['initialization_seconds']:.3f}s", flush=True)
    for label in ('first', 'warm'):
        result = reader.read(args.crop, profile=True)
        report['runs'].append({'pass': label, 'result': result})
        print(f"\n{label}: {result['seconds']:.3f}s; {len(result['lines'])} lines", flush=True)
        for name, seconds in result['timings']['stages_seconds'].items():
            print(f'  {name:42s} {seconds:9.4f}s', flush=True)
    output = args.output or Path(f'outputs/ocr_timing/stages_{args.precision}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
