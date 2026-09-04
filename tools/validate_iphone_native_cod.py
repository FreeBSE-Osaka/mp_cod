#!/usr/bin/env python3.11
"""Validate the physical-iPhone one-round native CoD artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cod_model


WEIGHT_SHA256 = "4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92"
STRUCTURE_MODEL = "mlx-community/Qwen3-0.6B-4bit"
BODY_MODEL = "mlx-community/Qwen3-1.7B-4bit"
PERSONAS = ["批判的設計者", "仮説構築者", "実証監査者", "実行設計者"]
MOVE_PREFIX = {
    "object": "その結論には異議があります。",
    "revise": "考え直しました。",
    "agree": "私もその案に賛成です。",
    "maintain": "結論は変わりません。",
}
REQUIRED_TERMS = {
    "PILOT": ["100人", "pilot", "修正率", "電池影響", "再評価"],
    "ROLLOUT": ["feature flag", "全利用者", "段階展開"],
    "DEVICE_SPLIT": ["新しい端末", "自動要約", "有効化", "旧端末", "既存経路"],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"INVALID: {message}")


def validate_position(position: dict, claims: dict[str, dict], data_ids: set[str]) -> None:
    code = position.get("claim")
    selected = position.get("data_ids")
    require(code in claims, f"unknown position claim: {code}")
    require(isinstance(selected, list) and selected, f"empty data_ids: {position.get('persona')}")
    require(len(selected) == len(set(selected)), f"duplicate data_ids: {position.get('persona')}")
    require(set(selected) <= set(claims[code]["supported_by"]), f"unsupported data_ids: {position.get('persona')}")
    require(set(selected) <= data_ids, f"unknown data_ids: {position.get('persona')}")
    require(isinstance(position.get("confidence"), int), "confidence must be an integer")
    require(1 <= position["confidence"] <= 100, "confidence is outside 1...100")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--maximum-seconds", type=float, default=35.0)
    parser.add_argument("--minimum-headroom-mib", type=float, default=512.0)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    require(result.get("schema_version") == 3, "schema_version must be 3")
    require(result.get("mode") == "native_cod_one_round", "unexpected mode")
    require(result.get("model") == STRUCTURE_MODEL, "unexpected structure model")
    require(result.get("body_model") == BODY_MODEL, "unexpected body model")
    require(result.get("adapter_weights_sha256") == WEIGHT_SHA256, "Weight SHA mismatch")
    require(result.get("hard_gate_pass") is True, "hard gate did not pass")
    start_thermal = result.get("start_thermal_state")
    require(start_thermal in {"nominal", "fair"}, "run started too hot")
    if start_thermal == "fair":
        require(result.get("body_adapter_load_required") is False, "fair start was used without a complete body cache")
    require(result.get("thermal_state") in {"nominal", "fair"}, "thermal state is too high")
    require(result.get("total_seconds", float("inf")) <= args.maximum_seconds, "runtime is too slow")
    require(
        result.get("minimum_limit_bytes_remaining", 0)
        >= int(args.minimum_headroom_mib * 1_048_576),
        "memory headroom is too small",
    )

    ledger = result.get("ledger")
    require(isinstance(ledger, dict), "ledger is missing")
    require(ledger.get("fixture_kind") == "synthetic_balanced", "fixture is not marked synthetic")
    data_ids = {row["id"] for row in ledger.get("data", [])}
    claims = {row["code"]: row for row in ledger.get("claims", [])}
    preferences = ledger.get("role_preferences", {})
    require(len(data_ids) == 5 and len(claims) == 3, "unexpected ledger size")
    require(set(preferences) == set(PERSONAS), "role_preferences persona mismatch")

    initial = result.get("initial_positions")
    reconciled = result.get("reconciliation_positions")
    require([row.get("persona") for row in initial or []] == PERSONAS, "initial persona order mismatch")
    require([row.get("persona") for row in reconciled or []] == PERSONAS, "reconciliation persona order mismatch")
    for row in initial:
        validate_position(row, claims, data_ids)
        require(row.get("origin") == "model", f"initial choice is not model-origin: {row['persona']}")
        require(row["claim"] in preferences[row["persona"]], f"initial choice is outside role_preferences: {row['persona']}")
    for row in reconciled:
        validate_position(row, claims, data_ids)

    initial_tally = dict(Counter(row["claim"] for row in initial))
    final_tally = dict(Counter(row["claim"] for row in reconciled))
    require(initial_tally == result.get("initial_tally"), "initial tally mismatch")
    require(final_tally == result.get("final_tally"), "final tally mismatch")
    require(len(initial_tally) >= 2, "blind choices collapsed to one claim")
    consensus = result.get("consensus_claim")
    outcome = result.get("outcome_status")
    if consensus is None:
        require(sorted(final_tally.values()) == [2, 2], "unresolved outcome is not a 2-to-2 tie")
        require(outcome == "unresolved_tie", "unresolved outcome status mismatch")
    else:
        require(consensus in claims and final_tally.get(consensus, 0) >= 3, "3-of-4 consensus is invalid")
        require(outcome == "consensus", "consensus outcome status mismatch")

    speakers = result.get("reconciliation_model_speakers")
    require(isinstance(speakers, list) and 0 < len(speakers) < 4, "reconciliation was not selective")
    require(result.get("retained_initial_votes") == 4 - len(speakers), "retained vote count mismatch")
    for row in reconciled:
        expected_origin = "model" if row["persona"] in speakers else "retained_initial"
        require(row.get("origin") == expected_origin, f"reconciliation origin mismatch: {row['persona']}")

    calls = result.get("structural_calls")
    require(isinstance(calls, list) and calls, "structural calls are missing")
    require(all(call.get("adapter_active") is False for call in calls), "LoRA was active during structural selection")
    require(result.get("structural_adapter_active") is False, "structural adapter flag is true")
    require(result.get("structural_repairs") == sum(not call.get("valid") for call in calls), "repair count mismatch")
    require(result.get("structural_repairs", 99) <= 1, "more than one structural repair was required")
    require(
        result.get("structural_json_extractions") == sum(bool(call.get("json_extracted")) for call in calls),
        "JSON extraction count mismatch",
    )
    require(
        {call["persona"] for call in calls if call["phase"] == "initial" and call["valid"]}
        == set(PERSONAS),
        "not every blind choice has a valid model call",
    )
    require(
        {call["persona"] for call in calls if call["phase"] == "reconciliation" and call["valid"]}
        == set(speakers),
        "reconciliation model-call speakers mismatch",
    )
    for call in calls:
        if not call.get("valid"):
            require(bool(call.get("validation_error")), f"invalid call lacks reason: {call['persona']}")
            continue
        parsed = cod_model.parse_json_object(call.get("raw_output", ""))
        require(isinstance(parsed, dict), f"raw structural JSON is invalid: {call['persona']}")
        if call["phase"] == "change_reason":
            require(set(parsed) == {"change_reason"}, f"change_reason schema mismatch: {call['persona']}")
            require(isinstance(parsed["change_reason"], str) and parsed["change_reason"], "empty change_reason")
            continue
        expected_keys = {"claim", "data_ids", "confidence"}
        require(set(parsed) == expected_keys, f"structural schema mismatch: {call['persona']}")
        validate_position(
            {
                "persona": call["persona"],
                "claim": parsed["claim"],
                "data_ids": parsed["data_ids"],
                "confidence": parsed["confidence"],
            },
            claims,
            data_ids,
        )

    changed_personas = {
        after["persona"]
        for before, after in zip(initial, reconciled)
        if before["claim"] != after["claim"]
    }
    reason_calls = {
        call["persona"]
        for call in calls
        if call["phase"] == "change_reason" and call["valid"]
    }
    require(reason_calls == changed_personas, "change_reason model calls do not match changed votes")
    for row in reconciled:
        require(
            bool(row.get("change_reason")) == (row["persona"] in changed_personas),
            f"saved change_reason mismatch: {row['persona']}",
        )
        if row["persona"] in changed_personas:
            reason = row["change_reason"]
            label = claims[row["claim"]]["label"]
            terms = REQUIRED_TERMS[row["claim"]]
            require(len(reason) <= min(60, len(label) + 24), f"change_reason is too long: {row['persona']}")
            require(
                sum(term in reason for term in terms) >= max(2, len(terms) - 1),
                f"change_reason lost the selected claim: {row['persona']}",
            )
            allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?", label))
            for item in ledger["data"]:
                if item["id"] in row["data_ids"]:
                    allowed_numbers.update(re.findall(r"\d+(?:\.\d+)?", item["text"]))
            require(
                set(re.findall(r"\d+(?:\.\d+)?", reason)) <= allowed_numbers,
                f"change_reason invented a number: {row['persona']}",
            )

    body_calls = result.get("body_calls")
    require(isinstance(body_calls, list), "body calls are missing")
    require(result.get("body_renderer_model_calls") == len(body_calls), "body model-call count mismatch")
    require(result.get("body_renderer_cache_hits") == 7 - len(body_calls), "body cache-hit count mismatch")
    require(result.get("body_fallbacks") == 0, "body fallback was used")
    require(
        result.get("body_politeness_sanitizations")
        == sum(bool(call.get("sanitized")) for call in body_calls),
        "body sanitization count mismatch",
    )
    for call in body_calls:
        require(call.get("valid") is True and call.get("fallback") is False, "invalid body call")
        label = claims[call["claim"]]["label"]
        require(cod_model.body_matches_claim(call["body"], label), f"body/claim mismatch: {call['claim']}")
        require(cod_model.body_is_neutral(call["body"]), f"body move leak: {call['claim']}")
        require(cod_model.body_is_polite_sentence(call["body"]), f"body is not polite: {call['claim']}")
        raw = cod_model.parse_json_object(call.get("raw_output", ""))
        parsed, warning, repaired = cod_model.parse_renderer_bodies(raw, ["B01"])
        require(warning is None and repaired is False, f"body schema repair: {call['claim']}")
        require(isinstance(parsed.get("B01"), str), f"body raw is missing: {call['claim']}")
        normalized, _ = cod_model.normalize_renderer_body(parsed["B01"])
        expected = (
            cod_model.sanitize_body_politeness(normalized, label)
            if call.get("sanitized")
            else normalized
        )
        require(expected == call["body"], f"saved body differs from validated raw: {call['claim']}")

    body_cache = result.get("persistent_body_cache")
    require(isinstance(body_cache, dict), "persistent body cache is missing")
    require(body_cache.get("schema_version") == 1, "body cache schema mismatch")
    require(body_cache.get("adapter_weights_sha256") == WEIGHT_SHA256, "body cache Weight SHA mismatch")
    entries = body_cache.get("entries")
    require(isinstance(entries, dict) and set(entries) == set(claims), "body cache entries mismatch")
    digest_rows = []
    for code in sorted(entries):
        entry = entries[code]
        require(entry.get("claim_label") == claims[code]["label"], f"cached claim label mismatch: {code}")
        raw = cod_model.parse_json_object(entry.get("raw_output", ""))
        parsed, warning, repaired = cod_model.parse_renderer_bodies(raw, ["B01"])
        require(warning is None and repaired is False, f"cached raw schema mismatch: {code}")
        normalized, _ = cod_model.normalize_renderer_body(parsed["B01"])
        expected = (
            cod_model.sanitize_body_politeness(normalized, entry["claim_label"])
            if entry.get("sanitized")
            else normalized
        )
        require(expected == entry.get("body"), f"cached body/raw mismatch: {code}")
        require(cod_model.body_matches_claim(entry["body"], entry["claim_label"]), f"cached body/claim mismatch: {code}")
        digest_rows.append(
            "\0".join(
                [
                    code,
                    entry["claim_label"],
                    entry["raw_output"],
                    entry["body"],
                    "true" if entry.get("sanitized") else "false",
                    entry["origin"],
                ]
            )
        )
    cache_digest = hashlib.sha256("\n".join(digest_rows).encode()).hexdigest()
    require(body_cache.get("source_payload_sha256") == cache_digest, "body cache digest mismatch")
    require(result.get("persistent_body_cache_entries") == 3, "body cache count mismatch")
    require(result.get("persistent_body_cache_source_sha256") == cache_digest, "result/cache digest mismatch")

    load_required = result.get("body_adapter_load_required")
    loaded = result.get("body_adapter_loaded")
    if load_required:
        require(loaded is True and len(body_calls) > 0, "required body Adapter was not loaded")
        require(result.get("body_adapter_loaded_after_structural_calls") is True, "body Adapter load order is unproven")
    else:
        require(loaded is False and len(body_calls) == 0, "body Adapter loaded despite complete cache")
        require(result.get("body_adapter_loaded_after_structural_calls") is False, "cached run reports Adapter load")
    require(result.get("adapter_unloaded") is True, "body Adapter was not unloaded")

    events = result.get("events")
    require(isinstance(events, list) and len(events) == 7, "expected four blind and three reaction events")
    require([event["id"] for event in events] == [f"C{i:02d}" for i in range(1, 8)], "event IDs are not sequential")
    require(all(event["move"] == "initial" for event in events[:4]), "blind event move mismatch")
    require([event["move"] for event in events[4:]] == ["object", "revise", "agree"], "reaction priority mismatch")
    for event in events:
        validate_position(event, claims, data_ids)
        require(re.search(r"D\d{2,}", event["utterance"]) is None, f"D ID leaked: {event['id']}")
        require(not event["body_origin"].startswith("fallback"), f"fallback event: {event['id']}")
        require(cod_model.body_matches_claim(event["body"], claims[event["claim"]]["label"]), f"event body mismatch: {event['id']}")
        if event["move"] in MOVE_PREFIX:
            require(event["utterance"] == MOVE_PREFIX[event["move"]] + event["body"], f"move/body composition mismatch: {event['id']}")
            require(event.get("target_event_id") in {"C01", "C02", "C03", "C04"}, f"reaction target mismatch: {event['id']}")
        else:
            require(event["utterance"] == event["body"], f"initial utterance mismatch: {event['id']}")

    repeat_identical = False
    if args.repeat:
        repeat = json.loads(args.repeat.read_text())
        semantic_keys = (
            "ledger",
            "initial_positions",
            "reconciliation_positions",
            "initial_tally",
            "final_tally",
            "consensus_claim",
            "outcome_status",
            "events",
            "reconciliation_model_speakers",
            "retained_initial_votes",
            "persistent_body_cache",
        )
        require(
            all(result.get(key) == repeat.get(key) for key in semantic_keys),
            "repeat run changed semantic output",
        )
        require(repeat.get("hard_gate_pass") is True, "repeat hard gate did not pass")
        require(repeat.get("thermal_state") in {"nominal", "fair"}, "repeat thermal state is too high")
        require(repeat.get("total_seconds", float("inf")) <= args.maximum_seconds, "repeat runtime is too slow")
        require(
            repeat.get("minimum_limit_bytes_remaining", 0)
            >= int(args.minimum_headroom_mib * 1_048_576),
            "repeat memory headroom is too small",
        )
        repeat_identical = True

    print(
        json.dumps(
            {
                "status": "valid",
                "initial_claims": len(initial_tally),
                "outcome": outcome,
                "events": len(events),
                "structural_calls": len(calls),
                "reconciliation_model_calls": len(speakers),
                "repeat_semantics_identical": repeat_identical,
                "body_calls": len(body_calls),
                "peak_footprint_mib": round(result["peak_footprint_bytes"] / 1_048_576, 3),
                "minimum_headroom_mib": round(result["minimum_limit_bytes_remaining"] / 1_048_576, 3),
                "total_seconds": round(result["total_seconds"], 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
