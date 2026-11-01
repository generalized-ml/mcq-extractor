#!/usr/bin/env python3
"""Compare MCQ JSON against golden JSON by question ID, without LLM calls."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


WEIGHTS = {
    "question": 0.1, "option_1": 0.2, "option_2": 0.2,
    "option_3": 0.2, "option_4": 0.2, "solution": 0.1,
}
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results" / "pdf_to_jsons"
LABELS = {str(i): i for i in range(1, 5)} | {c: i for i, c in enumerate("ABCD", 1)}
LLM_CALL_FIELDS = ("source", "response_id", "model", "status", "cost_usd",
                   "prompt_tokens", "completion_tokens", "total_tokens")


def llm_cost_report(payload: Any) -> tuple[list[dict], dict]:
    """Read recorded usage without counting checkpoint/question copies twice."""
    records = []
    hybrid = payload.get("hybrid") if isinstance(payload, dict) else None
    if isinstance(hybrid, dict) and isinstance(hybrid.get("results"), dict):
        # Includes incomplete candidates and calls whose questions are not in golden.
        records = list(hybrid["results"].items())
    else:
        buckets = {"questions": payload} if isinstance(payload, list) else payload
        if isinstance(buckets, dict):
            for bucket in ("questions", "review_candidates"):
                for index, question in enumerate(buckets.get(bucket, [])):
                    if isinstance(question.get("llm"), dict):
                        records.append((f"{bucket}/{index}", question["llm"]))

    def number(value: Any, *, integer: bool = False) -> int | float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (ValueError, TypeError, OverflowError):
            return None
        if not math.isfinite(result) or result < 0 or (integer and not result.is_integer()):
            return None
        return int(result) if integer else result

    calls, seen = [], {}
    for source, record in records:
        usage = record.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        call = {
            "source": source, "response_id": record.get("response_id") or "",
            "model": record.get("model") or "", "status": record.get("status", "recorded"),
            "cost_usd": number(usage.get("cost")),
            **{key: number(usage.get(key), integer=True) for key in
               ("prompt_tokens", "completion_tokens", "total_tokens")},
        }
        response_id = call["response_id"]
        if response_id and response_id in seen:
            previous = seen[response_id]
            if any(previous[key] != call[key] for key in LLM_CALL_FIELDS if key != "source"):
                raise ValueError(f"Conflicting usage for LLM response {response_id!r}")
            continue
        if response_id:
            seen[response_id] = call
        calls.append(call)
    costs = [call["cost_usd"] for call in calls if call["cost_usd"] is not None]
    summary = {
        "llm_recorded_calls": len(calls), "llm_priced_calls": len(costs),
        "llm_missing_cost_calls": len(calls) - len(costs),
        "llm_cost_usd": math.fsum(costs) if costs else None,
        "llm_cost_currency": "USD",
    }
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [call[key] for call in calls if call[key] is not None]
        summary[f"llm_{key}"] = sum(values) if values else None
    return calls, summary


def text_value(value: Any, location: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{location}: expected a string or null")
    return value


def field_text(record: dict, names: tuple[str, ...]) -> str:
    for name in names:
        if name in record:
            return text_value(record[name], name)
    return ""


def normalize_question(record: dict, position: int, *, strict: bool = False) -> dict:
    """Accept pipeline JSON and the simple text/options schema; preserve text."""
    if not isinstance(record, dict):
        raise ValueError(f"Question at position {position} must be an object")
    fields = dict.fromkeys(WEIGHTS, "")
    fields["question"] = field_text(record, ("question_text", "text", "question"))
    if "correct_answer" in record:
        answer = record["correct_answer"]
        if isinstance(answer, (str, int, float)) and not isinstance(answer, bool):
            fields["solution"] = str(answer)
        elif answer is not None:
            raise ValueError(f"Question {position}: correct_answer must be text, a number, or null")
    else:
        fields["solution"] = field_text(record, ("solution", "solution_text", "explanation"))
    warnings = []
    options = record.get("options", [])
    if not isinstance(options, list):
        raise ValueError(f"Question {position}: options must be an array")
    if len(options) > 4 and strict:
        raise ValueError(f"Question {position}: more than four options; resolve duplicates first")
    if len(options) != 4:
        warnings.append(f"Found {len(options)} options")
    used = set()
    for index, option in enumerate(options, 1):
        slot = index
        if isinstance(option, dict):
            if "label" in option:
                label = str(option["label"]).strip().strip("().").upper()
                if label not in LABELS:
                    if strict:
                        raise ValueError(f"Question {position}: unsupported option label {label!r}")
                    warnings.append(f"Unsupported option label {label!r}")
                    continue
                slot = LABELS[label]
            value = text_value(option.get("text"), f"Question {position}, option {slot}")
        else:
            value = text_value(option, f"Question {position}, option {slot}")
        if slot in used:
            if strict:
                raise ValueError(f"Question {position}: duplicate option label {slot}")
            warnings.append(f"Duplicate option label {slot}; ambiguous option scored as empty")
            fields[f"option_{slot}"] = ""
            continue
        used.add(slot)
        if slot <= 4:
            fields[f"option_{slot}"] = value
    identity = record.get("question_number")
    if identity is None or not str(identity).strip():
        identity = record.get("question_id", "")
    return {"identity": str(identity).strip() if identity is not None else "", "fields": fields,
            "schema_warnings": warnings}


def load_questions(path: Path, *, strict: bool = False) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("questions")
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON array or an object with a questions array")
    return [normalize_question(record, i, strict=strict) for i, record in enumerate(payload, 1)]


def keyword_match(prediction: str, golden: str) -> dict:
    """Exact matching prefix / golden word count. Empty golden text scores zero."""
    actual, expected = prediction.split(), golden.split()
    matched = 0
    for actual_word, expected_word in zip(actual, expected):
        if actual_word != expected_word:
            break
        matched += 1
    return {
        "score": matched / len(expected) if expected else 0.0,
        "matched_words": matched,
        "golden_words": len(expected),
        "prediction_words": len(actual),
        "first_mismatch_index": matched + 1 if matched < len(expected) else "",
        "expected_word": expected[matched] if matched < len(expected) else "",
        "actual_word": actual[matched] if matched < len(expected) and matched < len(actual) else "",
    }


def cosine_similarity(left: Any, right: Any) -> float:
    """Cosine in [-1, 1]; negative similarities are clipped to zero for scoring."""
    if len(left) != len(right) or not len(left):
        raise ValueError("Embedding vectors must have the same nonzero dimension")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm = math.sqrt(sum(float(a) ** 2 for a in left) * sum(float(b) ** 2 for b in right))
    value = dot / norm if norm else 0.0
    if not math.isfinite(value):
        raise ValueError("Embedding model returned non-finite values")
    return max(0.0, min(1.0, value))


class SentenceEncoder:
    def __init__(self, model: str, batch_size: int, local_files_only: bool,
                 device: str | None, revision: str | None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install evaluation dependencies with: python -m pip install -e '.[eval]'"
            ) from exc
        self.model = SentenceTransformer(
            model, device=device, revision=revision, local_files_only=local_files_only,
        )
        self.batch_size = batch_size

    def __call__(self, texts: list[str]) -> Any:
        # Split long fields into model-sized chunks instead of silently truncating.
        tokenizer = self.model.tokenizer
        special = tokenizer.num_special_tokens_to_add(pair=False)
        chunk_size = self.model.max_seq_length - special
        if chunk_size < 1:
            raise ValueError("Embedding model context is too small")
        chunks, groups = [], []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            indices = []
            for start in range(0, len(tokens), chunk_size):
                part = tokens[start:start + chunk_size]
                indices.append((len(chunks), len(part)))
                chunks.append(tokenizer.decode(part, skip_special_tokens=True))
            if not indices:
                indices.append((len(chunks), 1))
                chunks.append(text)
            groups.append(indices)
        vectors = self.model.encode(
            chunks, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        )
        return [
            sum(vectors[index] * weight for index, weight in group)
            / sum(weight for _, weight in group)
            for group in groups
        ]


def evaluate(predictions: list[dict], golden: list[dict],
             encode: Callable[[list[str]], Any]) -> tuple[list[dict], dict]:
    if not golden:
        raise ValueError("Golden dataset must contain at least one question")
    golden_ids = set()
    for expected in golden:
        identity = expected["identity"]
        if not identity:
            raise ValueError("Every golden question must have a question_number or question_id")
        if identity in golden_ids:
            raise ValueError(f"Duplicate golden question ID: {identity!r}")
        golden_ids.add(identity)
    predictions_by_id = {}
    for prediction in predictions:
        identity = prediction["identity"]
        if not identity:
            continue
        if identity in predictions_by_id:
            raise ValueError(f"Duplicate prediction question ID: {identity!r}; resolve ambiguity first")
        predictions_by_id[identity] = prediction
    aligned = [predictions_by_id.get(expected["identity"]) for expected in golden]
    # Only embed fields that can actually be compared; deduplicate repeated text.
    texts = {}
    for prediction, expected in zip(aligned, golden):
        if prediction is None:
            continue
        for name in WEIGHTS:
            left, right = prediction["fields"][name], expected["fields"][name]
            if left.strip() and right.strip():
                texts.setdefault(left, len(texts))
                texts.setdefault(right, len(texts))
    vectors = encode(list(texts)) if texts else []
    if len(vectors) != len(texts):
        raise ValueError("Embedding model returned the wrong number of vectors")
    rows = []
    for index, expected in enumerate(golden):
        prediction = aligned[index]
        row = {
            "position": index + 1,
            "golden_question_id": expected["identity"],
            "prediction_question_id": prediction["identity"] if prediction else "",
            "prediction_missing": prediction is None,
            "prediction_schema_warnings": "; ".join(prediction["schema_warnings"]) if prediction else "",
            "identifiers_match": prediction is not None,
        }
        for name in WEIGHTS:
            actual = prediction["fields"][name] if prediction else ""
            target = expected["fields"][name]
            detail = keyword_match(actual, target)
            row[f"{name}_keyword_score"] = detail.pop("score")
            row.update({f"{name}_{key}": value for key, value in detail.items()})
            row[f"{name}_semantic_score"] = (
                cosine_similarity(vectors[texts[actual]], vectors[texts[target]])
                if actual.strip() and target.strip() else 0.0
            )
            row[f"{name}_golden_empty"] = not bool(target.strip())
            row[f"{name}_prediction_empty"] = not bool(actual.strip())
        for metric in ("keyword", "semantic"):
            row[f"{metric}_score"] = sum(
                weight * row[f"{name}_{metric}_score"] for name, weight in WEIGHTS.items()
            )
        rows.append(row)
    summary = {
        "golden_count": len(golden), "prediction_count": len(predictions),
        "missing_predictions": sum(prediction is None for prediction in aligned),
        "extra_predictions": len(predictions) - sum(prediction is not None for prediction in aligned),
    }
    for name in WEIGHTS:
        summary[f"{name}_golden_empty_count"] = sum(row[f"{name}_golden_empty"] for row in rows)
        for metric in ("keyword", "semantic"):
            key = f"{name}_{metric}_score"
            summary[key] = sum(row[key] for row in rows) / len(rows)
    for metric in ("keyword", "semantic"):
        key = f"{metric}_score"
        summary[key] = sum(row[key] for row in rows) / len(rows)
    return rows, summary


def csv_values(row: dict) -> dict:
    return {key: f"{value:.8f}" if isinstance(value, float) else value for key, value in row.items()}


def save_results(output: Path, rows: list[dict], summary: dict, metadata: dict,
                 experiment_number: int | None, *,
                 llm_calls: list[dict] | None = None) -> tuple[int, Path]:
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "experiments.csv"
    with summary_path.open("a+", encoding="utf-8", newline="") as stream:
        # Serialize experiment numbering and append operations across local runs.
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        existing = list(csv.DictReader(stream))
        used = {int(row["experiment_number"]) for row in existing}
        number = experiment_number if experiment_number is not None else max(used, default=0) + 1
        if number in used:
            raise ValueError(f"Experiment {number} already exists; choose a new number")
        details_path = output / f"experiment_{number:04d}_questions.csv"
        metadata_path = output / f"experiment_{number:04d}_metadata.json"
        calls_path = output / f"experiment_{number:04d}_llm_calls.csv"
        if details_path.exists() or metadata_path.exists() or calls_path.exists():
            raise ValueError(f"Artifacts for experiment {number} already exist")
        record = {"experiment_number": number, **metadata, **summary}
        stream.seek(0)
        header = next(csv.reader(stream), None)
        if header is not None and header != list(record):
            raise ValueError("Existing experiments.csv has a different schema; use another output directory")
        with details_path.open("x", encoding="utf-8", newline="") as details:
            writer = csv.DictWriter(details, fieldnames=["experiment_number", *rows[0]])
            writer.writeheader()
            writer.writerows(csv_values({"experiment_number": number, **row}) for row in rows)
        if llm_calls is not None:
            with calls_path.open("x", encoding="utf-8", newline="") as calls_stream:
                writer = csv.DictWriter(calls_stream, fieldnames=LLM_CALL_FIELDS)
                writer.writeheader()
                # Preserve reported cost precision in the individual call ledger.
                writer.writerows(llm_calls)
        metadata_path.write_text(json.dumps({
            "experiment_number": number, **metadata, "weights": WEIGHTS,
            "llm_usage": {key: value for key, value in summary.items() if key.startswith("llm_")},
            "llm_cost_scope": "Reported usage.cost from saved extraction calls, deduplicated by response ID; includes incomplete and unmatched questions. Unknown costs are null, and totals cover only recorded values. Failed/retried attempts absent from the input are excluded. Evaluation makes no LLM calls.",
            "alignment": "exact question ID (question_number, falling back to question_id); output in golden order; missing IDs score zero",
            "keyword": "exact whitespace-token prefix / golden token count",
            "semantic": "cosine similarity clipped to [0,1]; token-weighted mean of model-sized chunks",
            "empty_fields": "zero for both metrics, including when both sides are empty; fixed weights",
            "aggregation": "mean over all golden questions; extra predictions reported separately",
            "images": "text only; images and image references are not compared",
            "solution_field": "correct_answer (string/number); falls back to solution/solution_text/explanation if absent",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stream.seek(0, 2)
        writer = csv.DictWriter(stream, fieldnames=list(record))
        if header is None:
            writer.writeheader()
        writer.writerow(csv_values(record))
    return number, details_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcq-json", required=True, type=Path)
    parser.add_argument("--golden-json", required=True, type=Path)
    parser.add_argument("--experiment-number", type=int, help="Defaults to the next unused experiment number")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model ID or local directory")
    parser.add_argument("--model-revision", help="Optional model commit for reproducible runs")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files")
    parser.add_argument("--device", help="Embedding device, e.g. cpu, mps, or cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.experiment_number is not None and args.experiment_number < 1):
        parser.error("Batch size and experiment number must be positive")
    try:
        prediction_path = args.mcq_json.expanduser().resolve()
        golden_path = args.golden_json.expanduser().resolve()
        predictions = load_questions(prediction_path)
        golden = load_questions(golden_path, strict=True)
        llm_calls, cost_summary = llm_cost_report(json.loads(prediction_path.read_text(encoding="utf-8")))
        if not golden:
            raise ValueError("Golden dataset is empty")
        metadata = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mcq_json": str(prediction_path), "golden_json": str(golden_path),
            "mcq_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "golden_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
            "model": args.model, "model_revision": args.model_revision or "default",
        }
        encoder = SentenceEncoder(args.model, args.batch_size, args.local_files_only,
                                  args.device, args.model_revision)
        rows, summary = evaluate(predictions, golden, encoder)
        summary.update(cost_summary)
        number, details = save_results(args.output_dir.expanduser().resolve(), rows, summary,
                                       metadata, args.experiment_number, llm_calls=llm_calls)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(f"Experiment: {number}")
    print(f"keyword_score: {summary['keyword_score']:.8f}")
    print(f"semantic_score: {summary['semantic_score']:.8f}")
    cost = summary["llm_cost_usd"]
    print(f"Recorded LLM cost (USD): {cost:.8f}" if cost is not None else "Recorded LLM cost (USD): unknown")
    print(f"Calls with reported cost: {summary['llm_priced_calls']}/{summary['llm_recorded_calls']}")
    print(f"LLM call costs: {details.parent / f'experiment_{number:04d}_llm_calls.csv'}")
    print(f"Per-question scores: {details}")
    print(f"Experiment scores: {details.parent / 'experiments.csv'}")
    if summary["solution_golden_empty_count"]:
        print(f"Note: {summary['solution_golden_empty_count']} golden answers are empty; "
              "their solution contribution is zero.")


if __name__ == "__main__":
    main()
