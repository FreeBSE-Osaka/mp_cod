#!/usr/bin/env python3.11
"""Audit measured GPU frames and CoD cancellation from a physical iPhone."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def phase_metrics(result, phase):
    interval = next(p for p in result["phases"] if p["name"] == phase)
    start, end = interval["started_at"], interval["ended_at"]
    require(finite_number(start) and finite_number(end) and end > start, f"invalid interval: {phase}")
    frames = [f for f in result["frames"] if f["phase"] == phase]
    require(all(start <= f["submitted_at"] <= f["completed_at"] <= end for f in frames),
            f"frames outside phase: {phase}")
    times = sorted(f["completed_at"] for f in frames)
    gaps = [right - left for left, right in zip([start, *times], [*times, end])]
    sorted_gaps = sorted(gaps)
    return {
        "frames": len(frames),
        "seconds": end - start,
        "fps": len(frames) / (end - start),
        "p95_gap_ms": sorted_gaps[max(0, math.ceil(len(gaps) * .95) - 1)] * 1000,
        "maximum_gap_ms": max(gaps) * 1000,
    }


def audit(result):
    require(result.get("schema_version") == 1, "unknown shadow schema")
    require(result.get("physical_device") is True, "not a physical-device run")
    require(result.get("production_integration_allowed") is False, "shadow cannot promote the app")
    require(result.get("input_kind") == "synthetic_fixed_cloud_fields_v1", "unknown renderer fixture")
    require(result.get("renderer") == "ExtremeWeather Regional3DMetalVolumeView", "unknown renderer")
    require((result.get("render_width"), result.get("render_height"), result.get("ray_steps"),
             result.get("target_fps")) == (384, 256, 24, 20), "render budget changed")
    for key in ("source_sha256", "shader_sha256"):
        require(isinstance(result.get(key), str) and len(result[key]) == 64
                and set(result[key]) <= set("0123456789abcdef"), f"invalid {key}")
    require(result.get("error") is None, "inference reported an error")
    require(result.get("frames"), "no rendered frames")
    require(all(f.get("succeeded") is True and finite_number(f.get("gpu_seconds"))
                and f["gpu_seconds"] > 0 and finite_number(f.get("submitted_at"))
                and finite_number(f.get("completed_at")) for f in result["frames"]),
            "failed GPU command or invalid timestamp")
    require([p["name"] for p in result["phases"]] == ["baseline_3d", "inference", "recovery_3d"],
            "missing or reordered phase")
    for before, after in zip(result["phases"], result["phases"][1:]):
        require(before["ended_at"] <= after["started_at"], "overlapping phases")
    require(result.get("samples"), "no memory or thermal samples")
    require(all(s["thermal"] in {"nominal", "fair"} for s in result["samples"]), "thermal gate failed")
    headroom = min(s["memory"]["limit_bytes_remaining"] for s in result["samples"]) / 1_048_576
    require(headroom >= 512, "memory headroom below 512 MiB")
    peak = max(s["memory"]["footprint_peak_bytes"] for s in result["samples"]) / 1_048_576
    released = [s["memory"] for s in result["samples"] if s["memory"]["stage"] == "after_inference_release"]
    require(len(released) == 1, "no unique post-inference release sample")
    require(released[0]["mlx_active_bytes"] < 1_048_576 and released[0]["mlx_cache_bytes"] == 0,
            "MLX buffers remained after inference")
    phases = {name: phase_metrics(result, name) for name in ("baseline_3d", "inference", "recovery_3d")}
    for name in ("baseline_3d", "recovery_3d"):
        require(phases[name]["seconds"] >= 3 and phases[name]["fps"] >= 15, f"3D frame rate too low: {name}")
        require(phases[name]["maximum_gap_ms"] <= 1000, f"3D stall: {name}")
    mode = result["mode"]
    latency = None
    if mode == "concurrent":
        require(result.get("cancelled") is False and result.get("stop_reason") is None, "concurrent run stopped")
        require(isinstance(result.get("cod_result"), dict), "missing CoD result")
        require(result["cod_result"].get("hard_gate_pass") is True, "CoD gate failed")
        require(phases["inference"]["fps"] >= 15, "concurrent 3D frame rate below 15 fps")
        require(phases["inference"]["p95_gap_ms"] <= 150
                and phases["inference"]["maximum_gap_ms"] <= 1000, "concurrent 3D stalled")
    else:
        expected = {"handoff": "3d_requested", "memory_warning_simulated": "memory_warning_simulated"}
        require(mode in expected, "unknown shadow mode")
        require(result.get("cancelled") is True and result.get("stop_reason") == expected[mode], "wrong cancellation reason")
        require(result.get("cod_result") is None, "cancelled run published a final CoD result")
        require(phases["inference"]["frames"] == 0, "handoff started 3D before MLX release")
        requested, finished = result.get("stop_requested_at"), result.get("inference_finished_at")
        require(finite_number(requested) and finite_number(finished), "missing cancellation clock")
        latency = finished - requested
        require(0 <= latency <= 3, "cancellation latency exceeded 3 seconds")
    return {"mode": mode, "status": "valid", "phases": phases, "peak_footprint_mib": peak,
            "minimum_headroom_mib": headroom, "cancellation_seconds": latency}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--shader-packet", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    packet = json.loads(args.shader_packet.read_text())
    try:
        require(hashlib.sha256(packet["shader"].encode()).hexdigest() == packet["shader_sha256"], "shader packet changed")
        require(all(result.get(k) == packet[k] for k in ("source_sha256", "shader_sha256")), "shader provenance mismatch")
        summary = audit(result)
        if result["mode"] == "concurrent":
            with tempfile.TemporaryDirectory(prefix="mp-cod-3d-audit-") as directory:
                nested = Path(directory) / "native.json"
                nested.write_text(json.dumps(result["cod_result"], ensure_ascii=False))
                native = Path(__file__).with_name("validate_iphone_native_cod.py")
                ledger = native.parents[1] / "data/typhoon18_20260825/native_cod_replay_ledger.json"
                command = [sys.executable, str(native), str(nested), "--ledger", str(ledger)]
                if args.reference:
                    command += ["--repeat", str(args.reference)]
                completed = subprocess.run(command, text=True, capture_output=True)
                require(completed.returncode == 0, completed.stdout + completed.stderr)
                summary["native_cod"] = json.loads(completed.stdout)
    except (ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"INVALID: {error}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
