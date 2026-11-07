"""Confidence limited to answer/solution plus three words."""
import math
import re

def line_confidence(lines):
    scores = [line["score"] for line in lines if line["text"].strip()]
    valid = bool(scores) and all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
    return {
        "score": (math.exp(sum(math.log(max(s, 1e-12)) for s in scores) / len(scores)) if valid else None),
        "low_token_fraction": sum(s < .5 for s in scores) / len(scores) if valid else None,
        "scored_lines": len(scores),
    }


def scope_confidence(parts):
    """Scope mobile confidence across crop continuations without trimming OCR text.

    Paddle returns line scores, so the boundary line contributes its original
    score even if the cutoff falls within it. No word confidence is invented.
    """
    text = "\n\n".join(part["text"] for part in parts)
    marker = re.search(r"\b(?:solution|answer)\b", text, re.IGNORECASE)
    end = len(text)
    if marker:
        # Use original offsets; punctuation-only tokens do not count as words.
        words = list(re.finditer(r"[^\s]+", text[marker.end():]))
        words = [word for word in words if re.search(r"\w", word.group())]
        end = marker.end() + words[min(2, len(words) - 1)].end() if words else marker.end()
    offset = 0
    selected = []
    for part in parts:
        local_end = max(0, min(len(part["text"]), end - offset))
        included, position = [], 0
        for line in part["lines"]:
            if position < local_end and line["text"].strip():
                included.append(line)
            position += len(line["text"]) + 1
        part["confidence"] = line_confidence(included)
        part["confidence_scope"] = {
            "rule": "through_first_solution_or_answer_plus_3_words",
            "marker_found": bool(marker), "text": part["text"][:local_end],
            "excluded": offset >= end and bool(marker),
            "granularity": "line_score_including_boundary_line",
        }
        if not part["confidence_scope"]["excluded"]:
            selected.append(part)
        offset += len(part["text"]) + 2
    return selected
