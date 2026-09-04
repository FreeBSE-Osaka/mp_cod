#!/usr/bin/env python3.11
"""Validate a physical-iPhone Claim Body soak result with production guards."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cod_model


EXPECTED_PERSONAS = {
    "力学モデル研究者",
    "アンサンブル確率予報者",
    "観測・ナウキャスト専門家",
    "影響・リスク予報者",
}
WEIGHT_SHA256 = "4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"INVALID: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cancel-result", type=Path)
    parser.add_argument("--minimum-headroom-mib", type=float, default=512.0)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    require(result.get("schema_version") == 2, "schema_version must be 2")
    require(result.get("mode") == "four_persona_body_soak", "unexpected mode")
    require(result.get("adapter_weights_sha256") == WEIGHT_SHA256, "Weight SHA mismatch")
    require(result.get("adapter_unloaded") is True, "Adapter was not unloaded")
    require(result.get("outputs_distinct") is True, "outputs_distinct is false")
    require(result.get("thermal_state") in {"nominal", "fair"}, "thermal state is too high")

    minimum_headroom = int(args.minimum_headroom_mib * 1_048_576)
    require(
        result.get("minimum_limit_bytes_remaining", 0) >= minimum_headroom,
        f"memory headroom is below {args.minimum_headroom_mib:g} MiB",
    )

    utterances = result.get("utterances")
    require(isinstance(utterances, list) and len(utterances) == 4, "expected four utterances")
    require({row.get("persona") for row in utterances} == EXPECTED_PERSONAS, "persona set mismatch")
    require(len({row.get("body") for row in utterances}) == 4, "duplicate bodies")

    exact_polite = 0
    for row in utterances:
        claim = row.get("claim")
        body = row.get("body")
        require(isinstance(claim, str) and isinstance(body, str), "claim/body must be strings")
        require(cod_model.body_matches_claim(body, claim), f"claim mismatch: {row.get('persona')}")
        require(cod_model.body_is_neutral(body), f"move leaked into body: {row.get('persona')}")
        require(cod_model.body_is_polite_sentence(body), f"body is not polite: {row.get('persona')}")
        require(re.search(r"D\d{2,}", body) is None, f"D ID leaked: {row.get('persona')}")

        raw = json.loads(row.get("raw_output", ""))
        parsed, warning, repaired = cod_model.parse_renderer_bodies(raw, ["B01"])
        require(warning is None and repaired is False, f"raw schema repair: {row.get('persona')}")
        require(parsed == {"B01": body}, f"raw/body mismatch: {row.get('persona')}")
        target = cod_model.sanitize_body_politeness(claim, claim)
        if target and body.rstrip("。！？!?") == target.rstrip("。！？!?"):
            exact_polite += 1

    cancel_valid = False
    if args.cancel_result:
        cancel = json.loads(args.cancel_result.read_text())
        require(cancel.get("schema_version") == 1, "cancel schema_version must be 1")
        require(cancel.get("adapter_unloaded") is True, "cancel path did not unload Adapter")
        require(cancel.get("completed_utterances", 4) < 4, "cancel path completed every utterance")
        require(cancel.get("thermal_state") in {"nominal", "fair"}, "cancel thermal state is too high")
        memory = cancel.get("memory_after_unload")
        require(isinstance(memory, dict), "cancel memory sample is missing")
        require(memory.get("mlx_cache_bytes") == 0, "cancel path left MLX cache allocated")
        cancel_valid = True

    peak_reduction_percent = None
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        baseline_utterances = baseline.get("utterances", [])
        require(
            [(row.get("persona"), row.get("body")) for row in baseline_utterances]
            == [(row.get("persona"), row.get("body")) for row in utterances],
            "baseline and optimized public bodies differ",
        )
        baseline_peak = baseline.get("peak_footprint_bytes", 0)
        optimized_peak = result.get("peak_footprint_bytes", 0)
        require(baseline_peak > optimized_peak > 0, "peak footprint did not improve")
        peak_reduction_percent = round((1 - optimized_peak / baseline_peak) * 100, 3)
        require(peak_reduction_percent >= 30, "peak footprint reduction is below 30%")

    print(
        json.dumps(
            {
                "status": "valid",
                "utterances": 4,
                "exact_polite": exact_polite,
                "peak_footprint_mib": round(result["peak_footprint_bytes"] / 1_048_576, 3),
                "minimum_headroom_mib": round(
                    result["minimum_limit_bytes_remaining"] / 1_048_576, 3
                ),
                "cancel_valid": cancel_valid,
                "peak_reduction_percent": peak_reduction_percent,
                "total_seconds": round(result["total_seconds"], 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
