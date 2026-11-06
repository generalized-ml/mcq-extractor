"""Run with: python -m pdf_to_json_my.model_quantization."""

import argparse
import json
from collections import Counter
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from . import MODEL_DIR
from ..ocr import DET_MODEL, DET_REVISION, REC_MODEL, REC_REVISION, cached_model


def convert_model(model_id, revision, output_dir, *, local_files_only=False):
    """Save a separate FP16 checkpoint; leave the source HF cache untouched."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    destination = output_dir / (model_id.rsplit('/', 1)[-1] + '_fp16')
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}; use a new --output-dir")
    source = Path(cached_model(model_id, revision, local_files_only=local_files_only))
    original = load_file(source / 'model.safetensors')
    print(f'Converting {model_id} from {source}', flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.fp16-', dir=output_dir) as temporary:
        temporary = Path(temporary)
        for asset in source.glob('*.json'):
            copy2(asset, temporary / asset.name)
        config_path = temporary / 'config.json'
        config = json.loads(config_path.read_text())
        config['dtype'] = 'float16'
        if 'torch_dtype' in config:
            config['torch_dtype'] = 'float16'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        tensors = {name: tensor.half() if tensor.is_floating_point() else tensor
                   for name, tensor in original.items()}
        with safe_open(source / 'model.safetensors', framework='pt') as checkpoint:
            metadata = checkpoint.metadata()
        save_file(tensors, temporary / 'model.safetensors', metadata=metadata)
        converted = load_file(temporary / 'model.safetensors')
        if original.keys() != converted.keys():
            raise ValueError('Conversion changed checkpoint tensor names')
        for name, tensor in original.items():
            expected = tensor.half() if tensor.is_floating_point() else tensor
            actual = converted[name]
            if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                raise ValueError(f'Incorrect conversion for {name}')
            if actual.is_floating_point() and not torch.isfinite(actual).all():
                raise ValueError(f'Non-finite FP16 values in {name}')
        report = {
            'source': model_id, 'revision': revision, 'source_directory': str(source),
            'dtype': 'float16', 'tensor_dtypes': dict(Counter(str(t.dtype) for t in converted.values())),
            'source_weight_bytes': (source / 'model.safetensors').stat().st_size,
            'fp16_weight_bytes': (temporary / 'model.safetensors').stat().st_size,
        }
        (temporary / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')
        temporary.rename(destination)
    print(json.dumps({'destination': str(destination), **report}, indent=2), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description='Save verified float16 OCR model copies locally.')
    parser.add_argument('--output-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    for model_id, revision in (
        (DET_MODEL, DET_REVISION),
        (REC_MODEL, REC_REVISION),
    ):
        convert_model(model_id, revision, args.output_dir.resolve(),
                      local_files_only=args.local_files_only)


if __name__ == '__main__':
    main()
