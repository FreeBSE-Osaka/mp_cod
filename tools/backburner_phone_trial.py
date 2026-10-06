#!/usr/bin/env python3.11
"""Bounded, localhost-only Backburner prefill trial; not distributed training."""
import argparse
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import urllib.request


def strict_json(data):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def non_finite(value):
        raise ValueError(f"non-finite JSON number: {value}")

    def finite_float(value):
        number = float(value)
        return number if math.isfinite(number) else non_finite(value)

    return json.loads(data, object_pairs_hook=unique_object, parse_constant=non_finite, parse_float=finite_float)


def post(port, path, payload):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(1_048_577)
    if len(data) > 1_048_576:
        raise ValueError("oversized localhost response")
    return strict_json(data)


def validate_phone_status(result):
    if not isinstance(result, dict):
        raise ValueError("phone status must be an object")
    thermal, available, chunks = (result.get(key) for key in ("thermal", "avail_mb", "tail_chunks"))
    if (type(thermal) is not int or thermal not in (0, 1)
            or type(available) not in (int, float) or not math.isfinite(available) or available < 512
            or type(chunks) is not int or chunks < 0 or type(result.get("cable_if_type")) is not int
            or result.get("cable_if_type") != 2 or result.get("wifi_paired") is not False
            or result.get("tail_state") not in {"ready", "connected", "busy"}):
        raise ValueError(f"phone resource, tail or cable-only gate failed: {result}")
    return result


def phone_status(address):
    if not isinstance(ipaddress.ip_address(address), ipaddress.IPv4Address) or not ipaddress.ip_address(address).is_link_local:
        raise ValueError("the experiment requires a wired IPv4 link-local phone address")
    with socket.create_connection((address, 50061), timeout=3) as connection:
        connection.sendall(b"mem\n")
        data = b""
        while b"\n" not in data and len(data) <= 65536:
            chunk = connection.recv(4096)
            if not chunk:
                break
            data += chunk
    if len(data) > 65536:
        raise ValueError("oversized phone status")
    return validate_phone_status(strict_json(data))


def offload_proof(before, after, log):
    # A connection or a Mac fallback is not a completed remote forward pass.
    if not all(type(row.get("tail_chunks")) is int and row["tail_chunks"] >= 0 for row in (before, after)):
        raise ValueError("missing or invalid phone chunk counter")
    chunks = after.get("tail_chunks", 0) - before.get("tail_chunks", 0)
    completed = sum(map(int, re.findall(r"split_finish: split (\d+) tokens done:", log)))
    fallback = bool(re.search(r"rerunning the batch locally|split prefill off|split prefill failed|merge of the worker's state failed", log))
    return {"phone_chunks": chunks, "merged_remote_tokens": completed,
            "fallback": fallback, "verified": chunks > 0 and completed > 0 and not fallback}


def complete_response(raw, prompt_tokens):
    tokens = raw.get("tokens")
    timings = raw.get("timings", {})
    return (raw.get("stop") is True and raw.get("stop_type") == "eos" and raw.get("truncated") is False
            and isinstance(raw.get("content"), str) and bool(raw["content"].strip())
            and isinstance(tokens, list) and 0 < len(tokens) <= 32
            and all(type(token) is int and token >= 0 for token in tokens)
            and type(raw.get("tokens_predicted")) is int and raw["tokens_predicted"] == len(tokens)
            and type(timings.get("cache_n")) is int and timings["cache_n"] == 0
            and type(timings.get("prompt_n")) is int and timings["prompt_n"] == prompt_tokens)


def prompt():
    records = "\n".join(f"資料{i:02d}: 効果は未測定です。計画はありますが、実行と成功は未確認です。" for i in range(18))
    return ("<|im_start|>system\n資料にない結果を創作せず、日本語で短く答えてください。<|im_end|>\n"
            f"<|im_start|>user\n{records}\n測定していない効果は確認済みと言えますか。"
            "いいえ、で始める短い一文だけを返してください。<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def run(args):
    import os
    if args.out.exists():
        raise ValueError("output directory already exists; do not overwrite a trial")
    if min(shutil.disk_usage(args.out.parent).free, shutil.disk_usage("/System/Volumes/Data").free) < 20 * 1024 ** 3:
        raise ValueError("20 GiB free-space guard failed")
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", args.port))
    args.out.mkdir()
    result = {"schema_version": 1, "mode": args.mode, "status": "running", "rows": [],
              "fixture_only": True, "training": False, "promotion_allowed": False,
              "usefulness_confirmed": False, "manual_semantic_review_required": True,
              "server": str(args.server), "model": str(args.model), "tail_layer": args.layer,
              "context": 1024, "batch": args.batch, "ubatch": args.ubatch, "seed": 20261006,
              "temperature": 0, "max_tokens": 32, "cache_prompt": False,
              "startup_and_warmup_excluded_from_rows": True}
    save = lambda: (args.out / "trial.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    save()
    process = None
    try:
        with args.model.open("rb") as stream:
            result["model_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        with args.server.open("rb") as stream:
            result["server_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("LLAMA_SPLIT_", "LLAMA_UBATCH_REMOTE", "PA_"))}
        if args.mode == "phone":
            result["phone_preflight"] = phone_status(args.phone)
            environment.update(LLAMA_SPLIT_TAIL=f"{args.phone}:50060", LLAMA_SPLIT_L=str(args.layer),
                               LLAMA_SPLIT_MIN="64", LLAMA_SPLIT_VERBOSE="1")
        command = [str(args.server), "-m", str(args.model), "-c", "1024", "-b", str(args.batch), "-ub", str(args.ubatch),
                   "-np", "1", "-ngl", "99", "--no-warmup", "--host", "127.0.0.1", "--port", str(args.port)]
        result["command"] = command
        with (args.out / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1) as health:
                        if strict_json(health.read(65536)).get("status") == "ok":
                            break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("server readiness deadline")
                time.sleep(0.2)
            text = prompt()
            tokens = post(args.port, "/tokenize", {"content": text})["tokens"]
            if not 192 <= len(tokens) <= 960:
                raise ValueError(f"fixture token count out of range: {len(tokens)}")
            result.update(prompt=text, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(), prompt_tokens=len(tokens))
            # ponytail: fixed synthetic read, no long-context agent service; expand only if this small real trial passes.
            for number in range(args.rounds + 1):
                before = phone_status(args.phone) if args.mode == "phone" else None
                offset = (args.out / "server.log").stat().st_size
                started = time.perf_counter()
                raw = post(args.port, "/completion", {"prompt": text, "temperature": 0, "seed": 20261006,
                           "n_predict": 32, "cache_prompt": False, "return_tokens": True})
                row = {"round": number, "warmup": number == 0, "wall_seconds": time.perf_counter() - started,
                       "raw": raw, "complete": complete_response(raw, len(tokens))}
                result["rows"].append(row)
                save()  # Preserve generated output even if remote proof fails afterwards.
                if args.mode == "phone":
                    after = phone_status(args.phone)
                    with (args.out / "server.log").open("rb") as evidence:
                        evidence.seek(offset)
                        segment = evidence.read().decode(errors="replace")
                    row.update(phone_before=before, phone_after=after, offload=offload_proof(before, after, segment))
                    save()
                    if not row["offload"]["verified"]:
                        raise RuntimeError("remote forward/merge proof failed; fallback is not success")
                if not row["complete"]:
                    raise RuntimeError("incomplete or truncated fixture output")
            result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)
        if process:
            result["server_exit_code"] = process.returncode
        save()
    print(json.dumps({"status": result["status"], "result": str(args.out / "trial.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("mac", "phone"), required=True)
    parser.add_argument("--phone", default="169.254.15.16")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--batch", type=int, choices=(256, 512), default=256)
    parser.add_argument("--ubatch", type=int, choices=(64, 128), default=64)
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--rounds", type=int, choices=(1, 2, 3), default=2)
    run(parser.parse_args())
