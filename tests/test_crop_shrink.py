"""Regression checks for reverse crop detection without macOS OCR or API calls."""

import unittest

from PIL import Image

from mcq_extractor.document.crop_layout import Box, Region, Segment
from mcq_extractor.document.crop_pipeline import content_floors, shrink_question


TEXT = "1. Pick a number\n(1) One\n(2) Two\n(3) Three\n(4) Four\nAnswer (2)\nSol. Printed explanation."


class Renderer:
    def crop(self, segment):
        return Image.new("RGB", (200, round(segment.box.bottom - segment.box.top)), "white")


def segment(page=1, column=0, top=0, bottom=300):
    box = Box(0, top, 200, bottom)
    return Segment(Region(page, column, box, 600, 800), box)


class ReverseCropTests(unittest.TestCase):
    def run_crop(self, parts=None, protected=None, ocr=None):
        self.calls = []

        def reader(image, *, recognition):
            self.calls.append(image.height)
            return {"text": TEXT, "mean_confidence": 1.0,
                    "lines": [{"text": TEXT, "bbox": [0, 10, 190, 80]}]}

        result = shrink_question("1", parts or [segment()], Renderer(), ocr or reader,
                                 step=24, require_answer=True, max_steps=100,
                                 min_confidence=.75, protected_bottoms=protected or [100], padding=4)
        self.addCleanup(result[2]["image"].close)
        return result

    def test_starts_full_and_trims_without_ocr_per_increment(self):
        parsed, parts, _, trace = self.run_crop()
        self.assertEqual(self.calls, [300, 108])
        self.assertEqual(parts[0].box.bottom, 108)
        self.assertEqual(trace[0]["phase"], "full_boundary")
        self.assertEqual(len(trace), 2)
        self.assertFalse(parsed["needs_review"])
        self.assertEqual(parsed["solution"], "Printed explanation.")

    def test_keeps_diagrams_below_last_ocr_line(self):
        _, parts, _, _ = self.run_crop(protected=[280])
        self.assertEqual(parts[0].box.bottom, 300)
        self.assertEqual(self.calls, [300, 300])

    def test_keeps_entire_solution_not_just_answer_marker(self):
        def reader(image, **kwargs):
            return {"text": TEXT, "mean_confidence": 1.0,
                    "lines": [{"text": "Last explanation line", "bbox": [0, 260, 190, 290]}]}
        _, parts, _, _ = self.run_crop(ocr=reader)
        self.assertEqual(parts[0].box.bottom, 300)

    def test_restores_full_crop_when_final_ocr_loses_content(self):
        calls = []

        def reader(image, **kwargs):
            calls.append(image.height)
            return {"text": TEXT if image.height == 300 else TEXT[:-10], "mean_confidence": 1.0,
                    "lines": [{"text": TEXT, "bbox": [0, 10, 190, 80]}]}

        parsed, parts, _, trace = self.run_crop(ocr=reader)
        self.assertEqual(calls, [300, 108, 300])
        self.assertEqual(parts[0].box.bottom, 300)
        self.assertEqual(trace[-1]["phase"], "restored_full_boundary")
        self.assertEqual(parsed["solution"], "Printed explanation.")

    def test_continuation_across_columns_and_pages_retains_segment_order(self):
        parts = [segment(1, 1), segment(2, 0)]

        def reader(image, **kwargs):
            return {"text": TEXT, "mean_confidence": 1.0,
                    "lines": [{"text": "continued solution", "bbox": [0, 330, 190, 396]}]}

        _, selected, _, _ = self.run_crop(parts=parts, protected=[300, 100], ocr=reader)
        self.assertEqual([(part.region.page, part.region.column) for part in selected], [(1, 1), (2, 0)])
        self.assertEqual([part.box.bottom for part in selected], [300, 108])

    def test_unreadable_structure_does_not_trigger_many_ocr_attempts(self):
        calls = []

        def reader(image, **kwargs):
            calls.append(image.height)
            return {"text": "1. unreadable diagram", "lines": [], "mean_confidence": .5}

        parsed, _, _, trace = self.run_crop(ocr=reader)
        self.assertTrue(parsed["needs_review"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(trace), 2)

    def test_docling_bottomleft_diagram_bounds_are_protected(self):
        raw = {"document": {"texts": [], "tables": [], "pictures": [
            {"prov": [{"page_no": 1, "bbox": {"l": 10, "r": 190,
              "t": 600, "b": 520, "coord_origin": "BOTTOMLEFT"}}]}]}}
        self.assertEqual(content_floors(raw, [segment()], padding=4), [284])


if __name__ == "__main__":
    unittest.main()
