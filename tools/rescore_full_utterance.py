#!/usr/bin/env python3.11
"""Rescore saved utterance raw into a new audit; never rewrite model outputs or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cod_model as cod
from tools.general_utterance_training import cases, example, profile_case, score


def rescore(evaluation, curated):
    source = json.loads(evaluation.read_text())
    curated_sha = hashlib.sha256(curated.read_bytes()).hexdigest()
    if source.get("curated_sha256") is not None and source["curated_sha256"] != curated_sha:
        raise ValueError("saved evaluation belongs to a different corpus")
    lookup = {case["case"]: case for case in cases(curated)}
    rows, seen = [], set()
    for row in source["results"]:
        key = (row["case"], row["profile"])
        if key in seen:
            raise ValueError("duplicate saved case/profile")
        seen.add(key)
        case = profile_case(lookup[row["case"]], row["profile"])
        if row.get("request") != example(case)["messages"][:2]:
            raise ValueError("saved request does not exactly match the corpus input")
        if any(field in row and row[field] != case[field] for field in ("claim", "move", "split", "speaker")):
            raise ValueError("saved case identity changed")
        current = score(row["raw"], case)
        rows.append({"case": key[0], "profile": key[1], "move": case["move"],
                     "raw_sha256": hashlib.sha256(row["raw"].encode()).hexdigest(),
                     "previous": {key: row.get(key) for key in ("strict_schema", "direct_valid", "study_valid")},
                     "current": current})
    return {"schema_version": 1, "mode": "saved_raw_rescore_not_new_generation",
            "evaluation": str(evaluation), "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
            "curated_sha256": curated_sha, "previous_validator_sha256": source.get("validator_sha256"),
            "current_validator_sha256": hashlib.sha256(Path(cod.__file__).read_bytes()).hexdigest(),
            "promotion_allowed": False, "selection_unchanged": True, "rows": rows,
            "identity_binding": "every_saved_request_exactly_matches_curated_case",
            "summary": {key: sum(bool(row["current"].get(key)) for row in rows)
                        for key in ("strict_schema", "direct_valid", "study_valid")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new audit output must not already exist")
    result = rescore(args.evaluation, args.curated)
    cod.write_json(args.out, result)
    print(json.dumps(result["summary"]))
