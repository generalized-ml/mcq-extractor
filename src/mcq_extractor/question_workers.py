"""Persistent question-level OCR workers; only the parent writes checkpoints."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp
import os
import time

from .crops import inside
from .ocr import HFMobileReader, cached_model

_reader = None
_initialization_seconds = None


def execution_settings(args, question_count):
    """Allocate one CPU worker per two physical cores, bounded by pending work."""
    import psutil
    if any(value is not None and value < 1 for value in (args.physical_cores, args.workers)):
        raise ValueError('Workers and physical core count must be positive')
    cores = args.physical_cores or psutil.cpu_count(logical=False)
    if not cores:
        raise ValueError('Cannot detect physical CPU cores; provide --physical-cores N')
    device = args.device
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu' and args.workers not in (None, 1):
        raise ValueError('Question multiprocessing requires --device cpu; use --workers 1 for GPU inference')
    requested = args.workers or (max(1, cores // 2) if device == 'cpu' else 1)
    workers = min(requested, question_count)
    return {'physical_cores': cores, 'workers': workers, 'threads_per_worker': min(2, cores),
            'device': device, 'question_count': question_count,
            'questions_per_worker_ceiling': (question_count + workers - 1) // workers if workers else 0,
            'scheduling': 'dynamic_whole_questions'}


def initialize_reader(options, threads, factory=None):
    global _reader, _initialization_seconds
    started = time.perf_counter()
    import torch
    import cv2
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    _reader = (factory or HFMobileReader)(**options)
    _initialization_seconds = time.perf_counter() - started


def transcribe_question(task):
    key, paths = task
    started = time.perf_counter()
    return {'key': key, 'transcripts': [_reader.read(path) for path in paths],
            'models': getattr(_reader, 'model_info', {}),
            'execution': {'worker_pid': os.getpid(), 'seconds': time.perf_counter() - started,
                          'model_initialization_seconds': _initialization_seconds}}


def transcribe_questions(items, root, options, settings, *, reader_factory=None):
    """Yield completed questions, with at most one in-flight task per worker."""
    if not items:
        return
    tasks = [(item['key'], [inside(root, part) for part in item['parts']]) for item in items]
    workers = settings['workers']
    if workers == 1:
        # Keep the single-worker escape hatch in process, including test readers.
        if reader_factory is None:
            import torch
            import cv2
            torch.set_num_threads(settings['threads_per_worker'])
            cv2.setNumThreads(1)
        started = time.perf_counter()
        reader = (reader_factory or HFMobileReader)(**options)
        initialization = time.perf_counter() - started
        for key, paths in tasks:
            started = time.perf_counter()
            transcripts = [reader.read(path) for path in paths]
            yield {'key': key, 'transcripts': transcripts, 'models': getattr(reader, 'model_info', {}),
                   'execution': {'worker_pid': os.getpid(), 'seconds': time.perf_counter() - started,
                                 'model_initialization_seconds': initialization}}
        return

    options = dict(options)
    if reader_factory is None:
        # Resolve/download once in the parent; spawned workers only read cache.
        for prefix in ('det', 'rec'):
            options[prefix + '_model'] = cached_model(
                options[prefix + '_model'], options[prefix + '_revision'],
                local_files_only=options['local_files_only'])
        options['local_files_only'] = True
    context = mp.get_context('spawn')
    iterator = iter(tasks)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=initialize_reader,
                             initargs=(options, settings['threads_per_worker'], reader_factory)) as executor:
        pending = {executor.submit(transcribe_question, next(iterator)) for _ in range(workers)}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                task = next(iterator, None)
                if task is not None:
                    pending.add(executor.submit(transcribe_question, task))
