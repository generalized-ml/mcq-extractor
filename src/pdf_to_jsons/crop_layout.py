"""Question anchors and one/two-column crop boundaries from Docling geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from .extract_mcqs import QUESTION, QUESTION_MARKER


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_docling(cls, value: dict, height: float) -> Box:
        if value.get("coord_origin", "TOPLEFT") == "BOTTOMLEFT":
            return cls(value["l"], height - value["t"], value["r"], height - value["b"])
        return cls(value["l"], value["t"], value["r"], value["b"])


@dataclass(frozen=True)
class Region:
    page: int
    column: int
    box: Box
    page_width: float
    page_height: float


@dataclass(frozen=True)
class Anchor:
    number: str
    region: int
    top: float
    source_ref: str


@dataclass(frozen=True)
class Segment:
    region: Region
    box: Box

    def to_dict(self) -> dict:
        return {"page": self.region.page, "column": self.region.column,
                "bbox": asdict(self.box), "coord_origin": "TOPLEFT",
                "page_width": self.region.page_width, "page_height": self.region.page_height}


def page_boxes(raw: dict, document: dict, page: int, height: float) -> list[Box]:
    """Prefer layout predictions, falling back to assembled item provenance."""
    pages = raw.get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    for entry in pages:
        if entry.get("page_no") != page:
            continue
        layout = (entry.get("predictions") or {}).get("layout") or {}
        boxes = [Box.from_docling(c["bbox"], height) for c in layout.get("clusters", [])
                 if c.get("label") not in {"page_header", "page_footer"} and c.get("bbox")]
        if boxes:
            return boxes
    return [Box.from_docling(prov["bbox"], height)
            for bucket in ("texts", "pictures", "tables") for item in document.get(bucket, [])
            if item.get("label") not in {"page_header", "page_footer"}
            for prov in item.get("prov", []) if prov["page_no"] == page]


def build_layout(raw: dict, *, padding: float = 4) -> tuple[list[Region], list[Anchor]]:
    document = raw.get("document", raw)
    regions = []
    for page_key, page in sorted(document["pages"].items(), key=lambda pair: int(pair[0])):
        page_no = int(page_key)
        width, height = page["size"]["width"], page["size"]["height"]
        boxes = page_boxes(raw, document, page_no, height)
        boxes = [b for b in boxes if b.right > b.left and b.bottom > b.top]
        if not boxes:
            continue
        # Search for a low-occupancy central gutter. Small formula/diagram boxes
        # must not bridge the gap, and full-width paragraphs veto two columns.
        substantial = [b for b in boxes if b.right - b.left >= width * .15]
        wide = sum(b.right - b.left > width * .7 for b in substantial)
        splits = []
        if substantial and wide / len(substantial) < .3:
            for fraction in range(35, 66):
                split = width * fraction / 100
                left_count = sum(b.right <= split for b in substantial)
                right_count = sum(b.left >= split for b in substantial)
                crossing = len(substantial) - left_count - right_count
                if min(left_count, right_count) >= 2 and crossing / len(substantial) < .15:
                    splits.append((crossing, abs(fraction - 50), split))
        if splits:
            split = min(splits)[2]
            boundaries = [(0, split), (split, width)]
        else:
            boundaries = [(0, width)]
        for column, (left, right) in enumerate(boundaries):
            members = [b for b in boxes if left <= (b.left + b.right) / 2 < right]
            if not members:
                continue
            regions.append(Region(page_no, column, Box(
                max(left, min(b.left for b in members) - padding),
                max(0, min(b.top for b in members) - padding),
                min(right, max(b.right for b in members) + padding),
                min(height, max(b.bottom for b in members) + padding)), width, height))

    candidates, content = [], []
    for item in document.get("texts", []):
        if item.get("label") in {"page_header", "page_footer"}:
            continue
        text, marker = item.get("text", "").strip(), item.get("marker", "").strip()
        standalone = QUESTION_MARKER.fullmatch(marker or text)
        numbered = standalone or (QUESTION.match(text) if not marker else None)
        stem = text if marker else ("" if standalone else numbered.group(2) if numbered else "")
        for prov in item.get("prov", []):
            for index, region in enumerate(regions):
                if region.page != prov["page_no"]:
                    continue
                box = Box.from_docling(prov["bbox"], region.page_height)
                if not region.box.left <= box.left < region.box.right:
                    continue
                content.append((index, box.top, f"{marker} {text}"))
                if numbered and box.left <= region.box.left + 32:
                    candidates.append((Anchor(numbered.group(1), index, box.top, item["self_ref"]), stem))
                break
    candidates.sort(key=lambda pair: (pair[0].region, pair[0].top))
    supported = set()
    for i, (anchor, stem) in enumerate(candidates):
        end = next((other.top for other, _ in candidates[i + 1:] if other.region == anchor.region), float("inf"))
        text = "\n".join(t for region, y, t in content if region == anchor.region and anchor.top <= y < end)
        labels = set(re.findall(r"\(([1-4A-Da-d])\)", text))
        if len(labels) >= 2 or re.search(r"\b(?:answer|ans\.)\s*[:.\-]?\s*\(?[1-4A-Da-d]\)?", text, re.I):
            supported.add(i)
    question_pages = {regions[candidates[i][0].region].page for i in supported}
    anchors, seen = [], set()
    for i, (anchor, stem) in enumerate(candidates):
        # Retain long stems on question pages even when their options were lost.
        if i not in supported and not (len(stem) >= 20 and regions[anchor.region].page in question_pages):
            continue
        key = (anchor.number, anchor.region, round(anchor.top, 1))
        if key not in seen:
            anchors.append(anchor)
            seen.add(key)
    return regions, anchors


def question_segments(regions: list[Region], anchors: list[Anchor], index: int,
                      *, padding: float = 4, max_regions: int = 4) -> tuple[list[Segment], bool]:
    anchor = anchors[index]
    following = anchors[index + 1] if index + 1 < len(anchors) else None
    last = following.region if following else len(regions) - 1
    truncated = last - anchor.region + 1 > max_regions
    last = min(last, anchor.region + max_regions - 1)
    segments = []
    for region_index in range(anchor.region, last + 1):
        region = regions[region_index]
        top = max(region.box.top, anchor.top - padding) if region_index == anchor.region else region.box.top
        bottom = min(region.box.bottom, following.top - padding) if following and region_index == following.region else region.box.bottom
        if bottom > top:
            segments.append(Segment(region, Box(region.box.left, top, region.box.right, bottom)))
    return segments, truncated
