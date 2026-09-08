#!/usr/bin/env python3.11
"""Reproducible General body-only corpus and paired direct-generation audit."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cod_model as cod


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases(path, splits):
    payload = json.loads(Path(path).read_text())
    names = [p["name"] for p in cod.load_domains()["general"]["personas"]]
    result, seen = [], set()
    for topic in payload["topics"]:
        if topic["id"] in seen or topic["split"] not in {"train", "valid", "test"}:
            raise ValueError("duplicate topic or invalid split")
        seen.add(topic["id"])
        for i, row in enumerate(topic["examples"]):
            if topic["split"] in splits:
                speaker = row.get("speaker", names[i % len(names)])
                if speaker not in names:
                    raise ValueError("unknown evaluation speaker")
                result.append({**row, "case": f"{topic['id']}:{i + 1}", "topic": topic["id"],
                               "split": topic["split"], "speaker": speaker})
    return result


def anchor_text(text):
    text = unicodedata.normalize("NFKC", text)
    for source, target in (("再度確認", "再確認"), ("当日の枠", "当日枠"),
                           ("日にだけ", "日だけ"), ("場合にだけ", "場合だけ"), ("のみ", "だけ")):
        text = text.replace(source, target)
    return re.sub(r"上限は(\d+(?:回\d+)?人)", r"\1まで", text)


def body_checks(body, case):
    label = case["claim"]
    normalized, reason = cod.normalize_renderer_body(body, label, speaker=case.get("speaker", ""))
    return {
        "body": normalized,
        "syntax": normalized is not None,
        "neutral": bool(normalized) and cod.body_is_neutral(normalized),
        "polite": bool(normalized) and cod.body_is_polite_sentence(normalized),
        "aligned": bool(normalized) and cod.body_matches_claim(normalized, label),
        "grounded": bool(normalized) and cod.dialogue_numbers_are_grounded(normalized, {"claim": label}),
        "anchors": bool(normalized) and all(re.search(anchor_text(p), anchor_text(normalized)) for p in case.get("required", [])),
        "competing": bool(normalized) and cod.dialogue_selects_competing_claim(normalized, case.get("competitors", [])),
        "reason": reason,
    }


def valid(checks):
    return all(checks[k] for k in ("syntax", "neutral", "polite", "aligned", "grounded", "anchors")) and not checks["competing"]


def score_output(raw, case):
    values, warning, repaired = cod.parse_renderer_bodies(cod.parse_json_object(raw), ["B01"])
    checks = body_checks(values.get("B01"), case)
    strict = warning is None and not repaired
    after = checks
    if checks["body"] and not checks["polite"]:
        polite = cod.sanitize_body_politeness(checks["body"], case["claim"])
        if polite:
            after = body_checks(polite, case)
    return {"checks": checks, "strict_schema": strict, "direct_valid": strict and valid(checks),
            "with_sanitizer_valid": strict and valid(after)}


def summarize(rows):
    return {split: {"total": len(group), "direct_valid": sum(r["direct_valid"] for r in group),
        "with_sanitizer_valid": sum(r["with_sanitizer_valid"] for r in group),
        "strict_schema": sum(r["strict_schema"] for r in group),
        "anchor_pass": sum(bool(r["checks"]["anchors"]) for r in group),
        "competing": sum(bool(r["checks"]["competing"]) for r in group)}
        for split in sorted({r["split"] for r in rows}) if (group := [r for r in rows if r["split"] == split])}


def legacy_cases(path):
    result = {}
    for row in json.loads(path.read_text())["results"]:
        result.setdefault(row["case"], {**row, "topic": "legacy_regression", "split": "legacy", "required": []})
    return list(result.values())


def rescore(args):
    if args.out.exists():
        raise ValueError("rescore output already exists")
    prior = json.loads(args.input.read_text())
    corpus = cases(args.curated, {"train", "valid", "test"})
    if args.legacy:
        corpus.extend(legacy_cases(args.legacy))
    by_case = {row["case"]: row for row in corpus}
    rows = []
    for row in prior["results"]:
        case = by_case[row["case"]]
        if row["claim"] != case["claim"]:
            raise ValueError("frozen claim changed since generation")
        case = {**case, "speaker": row["speaker"]}
        rows.append({"case": row["case"], "split": row["split"], **score_output(row["raw"], case)})
    result = {"schema_version": 1, "source": str(args.input), "source_sha256": sha(args.input),
              "curated_sha256": sha(args.curated), "validator_sha256": sha(cod.__file__),
              "evaluator_sha256": sha(__file__), "anchor_policy": "bounded_equivalences_v1",
              "policy": "saved raw only; no regeneration, no weight selection", "results": rows, "summary": summarize(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"]))


def example(claim, body, speaker, system=None):
    return {"messages": [
        {"role": "system", "content": cod.BODY_RENDERER_SYSTEM if system is None else system},
        {"role": "user", "content": json.dumps({"items": [{"id": "B01", "speaker": speaker, "claim": claim}]}, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps({"bodies": [{"id": "B01", "body": body}]}, ensure_ascii=False)},
    ]}


def build(args):
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise ValueError("dataset output already contains files")
    training_system = args.renderer_system_file.read_text().strip()
    if not training_system:
        raise ValueError("empty training renderer system")
    corpus = cases(args.curated, {"train", "valid", "test"})
    by_claim = {}
    for case in corpus:
        if case["claim"] in by_claim:
            raise ValueError("claim reused across examples/splits")
        by_claim[case["claim"]] = case["split"]
        checks = body_checks(case["body"], case)
        if not valid(checks):
            raise ValueError(f"invalid authored target {case['case']}: {checks}")
    speakers = [p["name"] for p in cod.load_domains()["general"]["personas"]]
    out.mkdir(parents=True, exist_ok=True)
    counts, hashes = {}, {}
    for split in ("train", "valid", "test"):
        rows = []
        for line in (args.rehearsal / f"{split}.jsonl").read_text().splitlines():
            old = json.loads(line)["messages"]
            item = json.loads(old[1]["content"])["items"][0]
            body = json.loads(old[2]["content"])["bodies"][0]["body"]
            if item["claim"] in by_claim:
                raise ValueError("new corpus overlaps rehearsal")
            rows.append(example(item["claim"], body, item["speaker"], training_system))
        old_count = len(rows)
        for case in corpus:
            if case["split"] == split:
                rows.extend(example(case["claim"], case["body"], speaker, training_system) for speaker in speakers)
        random.Random(20260908).shuffle(rows)
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        counts[split] = {"rehearsal": old_count, "curated": len(rows) - old_count, "total": len(rows)}
        hashes[split] = sha(path)
    manifest = {"schema_version": 1, "curated_sha256": sha(args.curated),
                "personas_sha256": sha(cod.PROFILE_PATH), "speakers": speakers,
                "topics": {s: sorted({c["topic"] for c in corpus if c["split"] == s}) for s in counts},
                "counts": counts, "sha256": hashes, "renderer_system": training_system,
                "policy": "authored synthetic targets plus v3 rehearsal; whole-topic split; no mined model outputs"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


def evaluate(args):
    check_isolation = getattr(args, "check_adapter_isolation", False)
    if check_isolation and not args.adapter:
        raise ValueError("adapter isolation check requires --adapter")
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import load_adapters, remove_lora_layers
    evaluation = cases(args.curated, set(args.split))
    if args.legacy:
        evaluation.extend(legacy_cases(args.legacy))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError("evaluation output already exists")
    model, tokenizer = load(str(args.model))
    probe = None
    if check_isolation:
        model.eval()
        first = evaluation[0]
        prompt = tokenizer.apply_chat_template(example(first["claim"], "unused", first["speaker"])["messages"][:2],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        before = generate(model, tokenizer, prompt=prompt, max_tokens=160, sampler=make_sampler(temp=0), verbose=False)
        probe = {"case": first["case"], "prompt": prompt, "base_before": before}
    if args.adapter:
        model = load_adapters(model, str(args.adapter))
    model.eval()
    mx.random.seed(20260908)
    result = {"schema_version": 1, "curated_sha256": sha(args.curated), "model": str(args.model),
              "renderer_system": cod.BODY_RENDERER_SYSTEM, "anchor_policy": "bounded_equivalences_v1",
              "validator_sha256": sha(cod.__file__), "evaluator_sha256": sha(__file__),
              "adapter": str(args.adapter) if args.adapter else None,
              "weights_sha256": sha(args.adapter / "adapters.safetensors") if args.adapter else None,
              "results": []}
    for case in evaluation:
        prompt = tokenizer.apply_chat_template(example(case["claim"], "unused", case["speaker"])["messages"][:2],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        start = time.perf_counter()
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=160, sampler=make_sampler(temp=0), verbose=False)
        row = {"case": case["case"], "topic": case["topic"], "split": case["split"],
               "claim": case["claim"], "speaker": case["speaker"], "raw": raw,
               **score_output(raw, case),
               "seconds": time.perf_counter() - start}
        result["results"].append(row)
        print(f"{case['case']} direct={row['direct_valid']} final={row['with_sanitizer_valid']} {row['checks']['body']}", flush=True)
        mx.clear_cache()
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    result["summary"] = summarize(result["results"])
    if probe is not None:
        model = remove_lora_layers(model)
        model.eval()
        after = generate(model, tokenizer, prompt=probe["prompt"], max_tokens=160, sampler=make_sampler(temp=0), verbose=False)
        result["adapter_isolation"] = {"case": probe["case"], "base_before": probe["base_before"],
            "base_after": after, "identical": probe["base_before"] == after,
            "scope": "one deterministic body probe; not full structural non-regression"}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "evaluate", "rescore"):
        p = sub.add_parser(name)
        p.add_argument("--curated", type=Path, default=Path(__file__).resolve().parents[1] / "data/general_body_v5/curated.json")
        p.add_argument("--out", type=Path, required=True)
        if name == "build":
            p.add_argument("--rehearsal", type=Path, required=True)
            p.add_argument("--renderer-system-file", type=Path, default=Path(__file__).resolve().parents[1] / "configs/claim-body-v5-system.txt")
        elif name == "evaluate":
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--adapter", type=Path)
            p.add_argument("--check-adapter-isolation", action="store_true")
            p.add_argument("--split", nargs="+", choices=("valid", "test"), default=["valid"])
            p.add_argument("--legacy", type=Path)
        else:
            p.add_argument("--input", type=Path, required=True)
            p.add_argument("--legacy", type=Path)
    args = parser.parse_args()
    {"build": build, "evaluate": evaluate, "rescore": rescore}[args.command](args)


if __name__ == "__main__":
    main()
