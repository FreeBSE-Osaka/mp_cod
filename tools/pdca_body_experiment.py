#!/usr/bin/env python3.11
"""Measure a PDCA system-prompt candidate with the existing body validator."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cod_model as cod
from tools import general_body_training as training
from tools.general_body_training import anchor_text, cases, example, score_output, sha


def model_identity(api_url, model):
    url = urllib.parse.urlsplit(api_url)._replace(path="/api/tags", query="", fragment="").geturl()
    with urllib.request.urlopen(url, timeout=15) as response:
        models = json.load(response).get("models", [])
    identity = next((row for row in models if model in (row.get("name"), row.get("model"))), None)
    if identity is None or not isinstance(identity.get("digest"), str):
        raise ValueError(f"installed model identity unavailable: {model}")
    return {key: identity.get(key) for key in ("name", "digest", "size", "details")}


def candidate_system(candidate, base_task=None):
    key = "additional_instruction" if base_task is not None else "system_prompt"
    if not isinstance(candidate, dict) or set(candidate) != {key}:
        raise ValueError(f"candidate must contain only {key}")
    system = candidate[key]
    source = {}
    if base_task is not None:
        if not isinstance(system, str) or len(system) > 400 or (system and not system.strip()):
            raise ValueError("additional_instruction must contain 0..400 characters")
        base_path = Path(base_task).resolve()
        base_bytes = base_path.read_bytes()
        base_data = json.loads(base_bytes)
        base_candidate = base_data.get("initial_candidate") if isinstance(base_data, dict) else None
        if not isinstance(base_candidate, dict) or set(base_candidate) != {"system_prompt"}:
            raise ValueError("base task must have an initial system_prompt candidate")
        base_system = base_candidate["system_prompt"]
        if not isinstance(base_system, str) or not base_system.strip():
            raise ValueError("base system_prompt must be nonempty")
        source = {"base_task_path": str(base_path),
                  "base_task_sha256": hashlib.sha256(base_bytes).hexdigest(),
                  "base_system_sha256": hashlib.sha256(base_system.encode()).hexdigest(),
                  "additional_instruction": system}
        system = base_system + ("\n追加規則: " + system if system.strip() else "")
    if not isinstance(system, str) or not 1 <= len(system.strip()) <= 2400:
        raise ValueError("system_prompt must contain 1..2400 characters")
    return system, source


def evaluate_candidate(args, request):
    system, source = candidate_system(request.get("candidate"), getattr(args, "base_task", None))
    expected_phase = "development" if args.split == "valid" else "holdout"
    if request.get("phase") != expected_phase:
        raise ValueError("experiment split does not match phase")
    output_dir = Path(request["output_dir"])
    if not output_dir.is_absolute() or not output_dir.is_dir():
        raise ValueError("output_dir must be an existing absolute directory")
    artifact = output_dir / "body_eval.json"
    if artifact.exists():
        raise ValueError("body evaluation artifact already exists")
    evaluation = cases(args.curated, {args.split})
    if not evaluation:
        raise ValueError("empty body evaluation split")
    identity = model_identity(args.api_url, args.model)
    audit = {
        "schema_version": 1, "phase": expected_phase, "model": identity,
        "api_url": args.api_url, "temperature": 0.0, "seed": args.seed,
        "num_predict": args.num_predict, "corpus_sha256": sha(args.curated),
        "validator_sha256": sha(cod.__file__), "evaluator_sha256": sha(__file__),
        "scorer_sha256": sha(training.__file__),
        "renderer_system": system, "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "results": [], **source,
    }
    cod.write_json(artifact, audit)
    started = time.perf_counter()
    for row in evaluation:
        messages = example(row["claim"], "unused", row["speaker"], system)["messages"]
        case_seed = cod.stable_seed(args.seed, row["speaker"], row["case"])
        raw, error, meta = "", None, {}
        try:
            _, meta = cod.ask_ollama(
                model=args.model, system=system, user=messages[1]["content"], schema="json",
                api_url=args.api_url, timeout=args.timeout, num_predict=args.num_predict,
                temperature=0.0, seed=case_seed,
                include_raw=True,
            )
            raw = meta.pop("_raw_content")
        except (RuntimeError, TimeoutError, ValueError) as exc:
            error = str(exc)
        scored = score_output(raw, row)
        body = scored["checks"]["body"]
        audit["results"].append({
            "case": row["case"], "claim": row["claim"], "speaker": row["speaker"],
            "split": row["split"], "topic": row["topic"], "seed": case_seed,
            "request": messages[:2], "raw": raw, "error": error, "generation": meta,
            **scored,
            "missing_anchors": [pattern for pattern in row.get("required", [])
                                if not body or not re.search(anchor_text(pattern), anchor_text(body))],
        })
        cod.write_json(artifact, audit)
    if model_identity(args.api_url, args.model)["digest"] != identity["digest"]:
        raise ValueError("installed model changed during measurement")
    rows = audit["results"]
    total = len(rows)
    count = sum(row["direct_valid"] for row in rows)
    metrics = {
        "total": total, "direct_valid": count, "direct_valid_rate": count / total,
        "with_sanitizer_valid": sum(row["with_sanitizer_valid"] for row in rows),
        "strict_schema": sum(row["strict_schema"] for row in rows),
        "generation_errors": sum(row["error"] is not None for row in rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "system_characters": len(system),
    }
    failures = [{
        "case": row["case"], "claim": row["claim"], "speaker": row["speaker"],
        "raw": row["raw"], "error": row["error"], "reason": row["checks"]["reason"],
        "missing_anchors": row["missing_anchors"],
        "failed_checks": [key for key in ("syntax", "neutral", "polite", "aligned", "grounded", "anchors")
                          if not row["checks"][key]]
                         + (["strict_schema"] if not row["strict_schema"] else [])
                         + (["competing"] if row["checks"]["competing"] else []),
    } for row in rows if not row["direct_valid"]]
    audit["metrics"] = metrics
    cod.write_json(artifact, audit)
    return {"schema_version": 1, "score": count / total, "passed": count == total,
            "metrics": metrics, "failures": failures, "artifacts": [str(artifact)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--base-task", type=Path,
                        help="Freeze this task's initial system_prompt; optimize additional_instruction only")
    parser.add_argument("--api-url", default=cod.DEFAULT_API_URL)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--num-predict", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.timeout <= 0 or args.num_predict <= 0:
        parser.error("timeout and num-predict must be positive")
    request_text = sys.stdin.read(1_000_001)
    if len(request_text) > 1_000_000:
        parser.error("experiment request too large")
    request = json.loads(request_text)
    if not isinstance(request, dict):
        parser.error("experiment request must be an object")
    print(json.dumps(evaluate_candidate(args, request), ensure_ascii=False))


if __name__ == "__main__":
    main()
