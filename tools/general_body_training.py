#!/usr/bin/env python3.11
"""Reproducible General body-only corpus and paired direct-generation audit."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import tempfile
import threading
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


def example(claim, body, speaker, system=None, *, constraint_hints=False, kind=None):
    item = cod.body_input_item("B01", speaker, claim, constraint_hints=constraint_hints, kind=kind)
    return {"messages": [
        {"role": "system", "content": cod.BODY_RENDERER_SYSTEM if system is None else system},
        {"role": "user", "content": json.dumps({"items": [item]}, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps({"bodies": [{"id": "B01", "body": body}]}, ensure_ascii=False)},
    ]}


def build(args):
    out = args.out
    constraint_hints = bool(getattr(args, "constraint_hints", False))
    rehearsal_limit = getattr(args, "rehearsal_train_limit", None)
    if rehearsal_limit is not None and rehearsal_limit < 1:
        raise ValueError("rehearsal train limit must be positive")
    if out.exists() and any(out.iterdir()):
        raise ValueError("dataset output already contains files")
    training_system = args.renderer_system_file.read_text().strip()
    if not training_system:
        raise ValueError("empty training renderer system")
    plain_path = getattr(args, "plain_system_file", None)
    plain_system = plain_path.read_text().strip() if plain_path else None
    if plain_path and (not constraint_hints or not plain_system):
        raise ValueError("plain-system-file requires constraint hints and a nonempty system")
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
            old_kind = item.get("rendering_hints", {}).get("claim_kind")
            old_kind = None if old_kind == "unspecified" else old_kind
            rows.append(example(item["claim"], body, item["speaker"], training_system,
                                constraint_hints=constraint_hints, kind=old_kind))
        if split == "train" and rehearsal_limit is not None:
            rows = random.Random(20260908).sample(rows, min(rehearsal_limit, len(rows)))
        old_count = len(rows)
        for case in corpus:
            if case["split"] == split:
                rows.extend(example(case["claim"], case["body"], speaker, training_system,
                                    constraint_hints=constraint_hints, kind=case.get("kind")) for speaker in speakers)
        source_count = len(rows)
        if plain_system is not None:
            plain_rows = []
            for row in rows:
                messages = row["messages"]
                item = json.loads(messages[1]["content"])["items"][0]
                body = json.loads(messages[2]["content"])["bodies"][0]["body"]
                plain_rows.append(example(item["claim"], body, item["speaker"], plain_system))
            rows.extend(plain_rows)
        random.Random(20260908).shuffle(rows)
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        factor = 2 if plain_system is not None else 1
        counts[split] = {"rehearsal": old_count * factor,
                         "curated": (source_count - old_count) * factor, "total": len(rows)}
        hashes[split] = sha(path)
    manifest = {"schema_version": 1, "curated_sha256": sha(args.curated),
                "personas_sha256": sha(cod.PROFILE_PATH), "speakers": speakers,
                "topics": {s: sorted({c["topic"] for c in corpus if c["split"] == s}) for s in counts},
                "counts": counts, "sha256": hashes, "renderer_system": training_system,
                "constraint_hints": constraint_hints,
                "plain_renderer_system": plain_system,
                "input_modes": ["source_hints", "plain"] if plain_system is not None else ["source_hints" if constraint_hints else "plain"],
                "policy": "authored synthetic targets plus v3 rehearsal; whole-topic split; no mined model outputs"}
    if rehearsal_limit is not None:
        manifest["policy"] = "authored synthetic targets plus frozen rehearsal; whole-topic split; no mined model outputs"
        manifest["rehearsal_sampling"] = {"train_limit": rehearsal_limit, "seed": 20260908,
            "source": str(args.rehearsal), "source_sha256": {s: sha(args.rehearsal / f"{s}.jsonl") for s in counts}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


def evaluate(args):
    constraint_hints = bool(getattr(args, "constraint_hints", False))
    system_path = getattr(args, "renderer_system_file", None)
    renderer_system = system_path.read_text().strip() if system_path else cod.BODY_RENDERER_SYSTEM
    if not renderer_system:
        raise ValueError("empty evaluation renderer system")
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
        prompt = tokenizer.apply_chat_template(example(first["claim"], "unused", first["speaker"], renderer_system,
            constraint_hints=constraint_hints, kind=first.get("kind"))["messages"][:2],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        before = generate(model, tokenizer, prompt=prompt, max_tokens=160, sampler=make_sampler(temp=0), verbose=False)
        probe = {"case": first["case"], "prompt": prompt, "base_before": before}
    if args.adapter:
        model = load_adapters(model, str(args.adapter))
    model.eval()
    mx.random.seed(20260908)
    result = {"schema_version": 1, "curated_sha256": sha(args.curated), "model": str(args.model),
              "renderer_system": renderer_system, "anchor_policy": "bounded_equivalences_v1",
              "constraint_hints": constraint_hints,
              "validator_sha256": sha(cod.__file__), "evaluator_sha256": sha(__file__),
              "adapter": str(args.adapter) if args.adapter else None,
              "weights_sha256": sha(args.adapter / "adapters.safetensors") if args.adapter else None,
              "results": []}
    for case in evaluation:
        messages = example(case["claim"], "unused", case["speaker"], renderer_system,
                           constraint_hints=constraint_hints, kind=case.get("kind"))["messages"][:2]
        prompt = tokenizer.apply_chat_template(messages,
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        start = time.perf_counter()
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=160, sampler=make_sampler(temp=0), verbose=False)
        row = {"case": case["case"], "topic": case["topic"], "split": case["split"],
               "claim": case["claim"], "speaker": case["speaker"], "raw": raw,
               "request": messages,
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


def nonthinking_tokenizer(tokenizer):
    # MLX forwards assignment and supplies True by default; override on the inner call.
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    template = inner.apply_chat_template
    def body_template(*values, **kwargs):
        kwargs["enable_thinking"] = False
        return template(*values, **kwargs)
    inner.apply_chat_template = body_template
    return tokenizer


def train(args):
    """Reuse MLX-LM's trainer, matching its prompt mask to our non-thinking inference."""
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("training adapter output already contains files")
    parent_adapter = getattr(args, 'parent_adapter', None)
    if parent_adapter is not None and not parent_adapter.is_file():
        raise ValueError("parent adapter must be an existing weights file")
    from mlx_lm import lora
    cache_mib = getattr(args, 'mlx_cache_limit_mib', None)
    if cache_mib is not None and (type(cache_mib) is not int or cache_mib < 0):
        raise ValueError('mlx-cache-limit-mib must be a nonnegative integer')
    if cache_mib is not None:
        import mlx.core as mx
    previous_cache = mx.set_cache_limit(cache_mib * 1024 * 1024) if cache_mib is not None else None
    original_load, original_argv = lora.load, sys.argv

    def load_body_model(*values, **kwargs):
        model, tokenizer = original_load(*values, **kwargs)
        return model, nonthinking_tokenizer(tokenizer)

    try:
        lora.load = load_body_model
        sys.argv = ["mlx_lm.lora", "--train", "--mask-prompt", "--model", str(args.model),
                    "--data", str(args.data), "--config", str(args.config),
                    "--adapter-path", str(args.out)]
        if parent_adapter is not None:
            sys.argv += ["--resume-adapter-file", str(parent_adapter)]
        lora.main()
    finally:
        lora.load, sys.argv = original_load, original_argv
        if cache_mib is not None:
            mx.set_cache_limit(previous_cache)
            print(json.dumps({'mlx_cache_limit_mib': cache_mib, 'MLX_peak_memory_GB': mx.get_peak_memory()/1e9,
                              'MLX_active_memory_GB': mx.get_active_memory()/1e9,
                              'MLX_cache_memory_GB': mx.get_cache_memory()/1e9}), flush=True)


def guarded_job(args, action):
    """Monitor this dedicated CLI process; keep a terminal record even on interruption."""
    from cod_pdca import _ResourceGuard, _finite
    if not _finite(args.min_free_gib) or args.min_free_gib < 0:
        raise ValueError("min-free-gib must be finite and nonnegative")
    if not _finite(args.resource_check_seconds) or args.resource_check_seconds <= 0:
        raise ValueError("resource-check-seconds must be finite and positive")
    if args.min_free_gib and threading.current_thread() is not threading.main_thread():
        raise ValueError("resource-monitored body jobs must run on the main thread of a dedicated CLI process")
    journal = args.out.with_name(args.out.name + ".job.json")
    if journal.exists():
        raise ValueError("job journal already exists; use a new output path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    guard = _ResourceGuard([Path.cwd(), args.out.parent, tempfile.gettempdir()],
                           args.min_free_gib, args.resource_check_seconds)
    started = time.monotonic()
    state = {"schema_version": 1, "status": "running", "command": args.command,
             "argv": sys.argv, "source_sha256": sha(__file__),
             "training_enable_thinking": False if args.command == "train" else None,
             "min_free_gib": args.min_free_gib,
             "resource_check_seconds": args.resource_check_seconds}
    cod.write_json(journal, state)
    error = None
    try:
        guard.start()
        action(args)
    except BaseException as caught:
        error = caught
    finally:
        try:
            guard.close()
        except KeyboardInterrupt as caught:
            error = caught
            guard.close()
    if guard.reason:
        state.update(status="resource_stopped", resource_stop=guard.reason)
        code = 2
    elif isinstance(error, KeyboardInterrupt):
        state.update(status="interrupted", error="KeyboardInterrupt")
        code = 130
    elif isinstance(error, SystemExit):
        code = error.code if type(error.code) is int else 0 if error.code is None else 1
        state.update(status="completed" if code == 0 else "failed", error=str(error), error_type="SystemExit")
    elif error is not None:
        state.update(status="failed", error=str(error), error_type=type(error).__name__)
        code = 1
    else:
        state["status"] = "completed"
        code = 0
    state.update(elapsed_seconds=round(time.monotonic() - started, 3), returncode=code,
                 resource_observation=guard.latest)
    cod.write_json(journal, state)
    if error is not None and code == 1:
        raise error.with_traceback(error.__traceback__)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "evaluate", "rescore", "train"):
        p = sub.add_parser(name)
        p.add_argument("--curated", type=Path, default=Path(__file__).resolve().parents[1] / "data/general_body_v5/curated.json")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--min-free-gib", type=float, default=20.0)
        p.add_argument("--resource-check-seconds", type=float, default=2.0)
        if name == "build":
            p.add_argument("--constraint-hints", action="store_true")
            p.add_argument("--plain-system-file", type=Path,
                           help="Duplicate the same sources/targets with plain inputs and this system for joint rehearsal")
            p.add_argument("--rehearsal", type=Path, required=True)
            p.add_argument("--rehearsal-train-limit", type=int)
            p.add_argument("--renderer-system-file", type=Path, default=Path(__file__).resolve().parents[1] / "configs/claim-body-v5-system.txt")
        elif name == "evaluate":
            p.add_argument("--constraint-hints", action="store_true")
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--adapter", type=Path)
            p.add_argument("--renderer-system-file", type=Path)
            p.add_argument("--check-adapter-isolation", action="store_true")
            p.add_argument("--split", nargs="+", choices=("valid", "test"), default=["valid"])
            p.add_argument("--legacy", type=Path)
        elif name == "rescore":
            p.add_argument("--input", type=Path, required=True)
            p.add_argument("--legacy", type=Path)
        else:
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--data", type=Path, required=True)
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--parent-adapter", type=Path,
                           help="Resume these LoRA weights; omit to initialize a new Adapter from Base")
            p.add_argument("--mlx-cache-limit-mib", type=int,
                           help="Training-only MLX allocator cache cap; 0 disables unused-memory caching")
    args = parser.parse_args()
    return guarded_job(args, {"build": build, "evaluate": evaluate, "rescore": rescore, "train": train}[args.command])


if __name__ == "__main__":
    raise SystemExit(main())
