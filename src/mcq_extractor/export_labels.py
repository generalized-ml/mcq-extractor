"""Export detected line crops for human correction before recognition fine-tuning."""
import argparse
import json
from pathlib import Path

from .crops import inside
from .ocr import rectify
from mcq_extractor.common import file_hash


def export(run_dir, output):
    import numpy as np
    from PIL import Image
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    state = json.loads((run_dir / 'pipeline.json').read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError('Label export directory must be empty to protect human corrections')
    output.mkdir(parents=True, exist_ok=True)
    images = output / 'images'
    images.mkdir(exist_ok=True)
    count = 0
    with (output / 'labels.jsonl').open('w', encoding='utf-8') as manifest:
        for item in state['plan']:
            record = state['results'].get(item['key'])
            if not record:
                continue
            ocr_path = f"questions/{item['key']}/paddle_ocr.json"
            if file_hash(inside(run_dir, ocr_path)) != record['artifacts'][ocr_path]:
                raise ValueError('OCR artifact changed since checkpoint')
            parts = json.loads(inside(run_dir, ocr_path).read_text())['parts']
            if len(parts) != len(item['parts']):
                raise ValueError('OCR and crop part counts differ')
            for part_index, (relative, part) in enumerate(zip(item['parts'], parts)):
                if file_hash(inside(run_dir, relative)) != item['artifacts'][relative]:
                    raise ValueError('Crop changed since checkpoint')
                with Image.open(inside(run_dir, relative)) as source:
                    pixels = np.asarray(source.convert('RGB'))
                for line_index, line in enumerate(part['lines']):
                    name = f"{item['key']}_{part_index:02}_{line_index:04}.png"
                    crop = rectify(pixels, line['box'])
                    crop.save(images / name)
                    crop.close()
                    manifest.write(json.dumps({
                        'image': f'images/{name}', 'suggested_text': line['text'],
                        'text': None, 'verified': False, 'score': line['score'],
                        'source_pdf_sha256': state['identity']['pdf_sha256'],
                        'question_number': item['question_number'], 'source_crop': relative,
                        'box': line['box'],
                    }, ensure_ascii=False) + '\n')
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(f'Exported {export(args.run_dir, args.output)} line crops for human labeling')


if __name__ == '__main__':
    main()
