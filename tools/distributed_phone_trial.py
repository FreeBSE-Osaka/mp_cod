#!/usr/bin/env python3.11
"""Bounded same-model Mac/physical-iPhone job splitting experiment (not layer splitting)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cod_model as cod

MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"
ADAPTER_SHA = "4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92"
BUNDLE_ID = "com.freebse.MPCoDClaimBodyHarness"
SYSTEM = '各itemのspeakerとして、検証済みclaimを自然な日本語一文で述べる本文renderer。claimの内容、時制、数字を変更・追加せず、moveや賛否は表現しない。入力itemsと同じidを一度ずつ返す。出力はbodiesだけをキーに持つJSONで、各要素のキーはidとbodyだけ。必ず {"bodies":[{"id":"入力id","body":"本文"}]} の形で返し、idをJSONキーにしてはならない。提案や計画を実現済み・検証済み等の完了事実へ変えず、claimの時制と確実性を保つ。'
JOBS = (
    ("力学モデル研究者", "進路予測は上層場と地上場の整合を確認して更新する", ["進路予測", "上層場", "地上場", "更新"]),
    ("アンサンブル確率予報者", "少数だが重大なシナリオも分布に残して比較する", ["少数", "重大", "シナリオ", "比較"]),
    ("観測・ナウキャスト専門家", "根拠データの観測時刻と出典を毎回確認する", ["観測時刻", "出典", "確認"]),
    ("影響・リスク予報者", "暴風が強まる前の安全確保を優先する", ["暴風", "安全確保", "優先"]),
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_body(raw, index):
    parsed, warning, repaired = cod.parse_renderer_bodies(json.loads(raw), ["B01"])
    if warning or repaired or set(parsed) != {"B01"}:
        raise ValueError(f"job {index}: raw JSON contract failed")
    body = parsed["B01"]
    _, claim, terms = JOBS[index]
    if (not all(term in body for term in terms) or not cod.body_matches_claim(body, claim)
            or not cod.body_is_neutral(body) or not cod.body_is_polite_sentence(body)
            or re.search(r"D\d{2,}", body)):
        raise ValueError(f"job {index}: body contract failed")
    return body


def validate_phone(result, request_id, indices, model_sha):
    if (result.get("schema_version") != 3 or result.get("mode") != "distributed_body_worker"
            or result.get("distributed_request_id") != request_id
            or result.get("distributed_job_indices") != indices
            or result.get("model_weights_sha256") != model_sha
            or result.get("adapter_weights_sha256") != ADAPTER_SHA
            or result.get("adapter_unloaded") is not True
            or result.get("outputs_distinct") is not True
            or result.get("thermal_state") not in {"nominal", "fair"}
            or result.get("minimum_limit_bytes_remaining", 0) < 512 * 1_048_576):
        raise ValueError("phone result identity, lifecycle or resource gate failed")
    if not any(sample.get("stage") == "after_adapter_unload" and sample.get("mlx_cache_bytes") == 0
               for sample in result.get("memory_samples", [])):
        raise ValueError("phone did not record cache release after unloading")
    rows = result.get("utterances")
    if not isinstance(rows, list) or len(rows) != len(indices):
        raise ValueError("phone returned the wrong number of jobs")
    for index, row in zip(indices, rows):
        persona, claim, _ = JOBS[index]
        if row.get("persona") != persona or row.get("claim") != claim:
            raise ValueError("phone returned an unexpected job")
        if validate_body(row.get("raw_output", ""), index) != row.get("body"):
            raise ValueError("phone raw/body mismatch")
        if not 0 < row.get("generation_tokens", 0) <= 96:
            raise ValueError("phone did not generate within the token budget")
    return rows


def command(arguments, receipt):
    # CoreDevice JSON, not display tables, is the machine-readable command receipt.
    completed = subprocess.run(["xcrun", "devicectl", *arguments[:3],
                                "--json-output", str(receipt), *arguments[3:]],
                               capture_output=True, text=True, timeout=45)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed


def phone_jobs(device, indices, directory, model_sha):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    command(["device", "process", "launch", "--device", device, "--terminate-existing",
             BUNDLE_ID, "--", "--autorun", "--distributed-worker", "--request-id", request_id,
             "--job-indices", ",".join(map(str, indices))], directory / "launch.json")
    destination = directory / "phone_result.json"
    deadline = time.monotonic() + 90
    last_error = None
    for attempt in range(45):
        if time.monotonic() >= deadline:
            break
        try:
            command(["device", "copy", "from", "--device", device,
                     "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE_ID,
                     "--source", f"Documents/mp_cod_distributed_{request_id}.json",
                     "--destination", str(destination)], directory / f"retrieve_{attempt}.json")
            result = json.loads(destination.read_text())
            validate_phone(result, request_id, indices, model_sha)
            return {"wall_seconds": time.perf_counter() - started, "request_id": request_id,
                    "result": result}
        except RuntimeError as error:
            last_error = str(error)
            time.sleep(1)
    raise RuntimeError(f"phone result timeout: {last_error}")


def mac_jobs(model, tokenizer, indices, checkpoint=None):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    rows = []
    started = time.perf_counter()
    for index in indices:
        persona, claim, _ = JOBS[index]
        user = json.dumps({"items": [{"id": "B01", "speaker": persona, "claim": claim}]},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        prompt = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                               {"role": "user", "content": user}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
        mx.random.seed(20261006)
        job_start = time.perf_counter()
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=96, sampler=make_sampler(temp=0), verbose=False)
        row = {"index": index, "persona": persona, "claim": claim,
               "raw_output": raw, "body": None, "valid": False,
               "seconds": time.perf_counter() - job_start,
               "prompt_tokens": len(tokenizer.encode(prompt))}
        try:
            row.update(body=validate_body(raw, index), valid=True)
        except (ValueError, TypeError, KeyError) as error:
            row["validation_error"] = str(error)
        rows.append(row)
        if checkpoint:
            cod.write_json(checkpoint, {"rows": rows, "complete": len(rows) == len(indices)})
        mx.clear_cache()
    return {"wall_seconds": time.perf_counter() - started, "rows": rows,
            "all_valid": all(row["valid"] for row in rows)}


def compare(args):
    if args.out.exists():
        raise ValueError("trial directory already exists")
    if digest(args.adapter / "adapters.safetensors") != ADAPTER_SHA:
        raise ValueError("unexpected legacy Claim Body adapter")
    if min(shutil.disk_usage(args.out.parent).free, shutil.disk_usage("/System/Volumes/Data").free) < 20 * 1024 ** 3:
        raise ValueError("20 GiB free-space guard failed")
    model_sha = digest(args.model / "model.safetensors")
    args.out.mkdir()
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters
    model, tokenizer = load(str(args.model))
    model = load_adapters(model, str(args.adapter)); model.eval()
    warmup = mac_jobs(model, tokenizer, [0], args.out / "mac_warmup.json")
    result = {"schema_version": 1, "experiment": "independent_body_jobs_not_layer_splitting",
              "fixture_only": True, "model": MODEL_ID, "model_weights_sha256": model_sha,
              "adapter_weights_sha256": ADAPTER_SHA, "max_tokens": 96, "temperature": 0,
              "warmup": warmup, "rounds": [], "usefulness_confirmed": False,
              "promotion_allowed": False}
    cod.write_json(args.out / "comparison.json", result)
    # ponytail: two jobs per worker, foreground launches; persistent authenticated workers only after measured benefit.
    with ThreadPoolExecutor(max_workers=1) as executor:
        for number in range(args.rounds):
            directory = args.out / f"round_{number + 1}"; directory.mkdir()
            baseline = mac_jobs(model, tokenizer, [0, 1, 2, 3], directory / "mac_only.json")
            started = time.perf_counter()
            remote = executor.submit(phone_jobs, args.device, [1, 3], directory, model_sha)
            local = mac_jobs(model, tokenizer, [0, 2], directory / "distributed_mac.json")
            phone = remote.result()
            wall = time.perf_counter() - started
            result["rounds"].append({"mac_only": baseline, "distributed_mac": local,
                                     "distributed_phone": phone, "distributed_wall_seconds": wall,
                                     "speedup": baseline["wall_seconds"] / wall,
                                     "quality_gate_pass": baseline["all_valid"] and local["all_valid"],
                                     "phone_launch_and_retrieval_included": True,
                                     "first_time_model_transfer_excluded": True})
            cod.write_json(args.out / "comparison.json", result)
    print(json.dumps({"rounds": len(result["rounds"]),
                      "speedups": [r["speedup"] for r in result["rounds"]],
                      "result": str(args.out / "comparison.json")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, choices=(1, 2, 3), default=2)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
