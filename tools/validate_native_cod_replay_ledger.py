#!/usr/bin/env python3.11
"""Validate a historical real-data ledger before bundling it into iOS."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cod_model


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"INVALID: {message}")


def parse_time(value: object, field: str) -> dt.datetime:
    require(isinstance(value, str), f"{field} must be a string")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, f"{field} must include a timezone")
    return parsed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text())
    require(ledger.get("schema_version") == 1, "schema_version must be 1")
    require(ledger.get("fixture_kind") == "historical_real_data_snapshot", "fixture kind mismatch")
    require(ledger.get("historical_replay") is True, "historical_replay must be true")
    cutoff = parse_time(ledger.get("cutoff"), "cutoff")
    require("歴史的" in ledger.get("topic", ""), "topic must state the historical boundary")

    provenance = ledger.get("provenance")
    require(isinstance(provenance, dict), "provenance is missing")
    require(provenance.get("live_retrieval") is False, "replay must not claim live retrieval")
    for path_key, hash_key in (
        ("parent_ledger", "parent_ledger_sha256"),
        ("data_packet", "data_packet_sha256"),
    ):
        path = args.repo_root / provenance[path_key]
        require(path.is_file(), f"missing provenance file: {path}")
        require(sha256(path) == provenance[hash_key], f"provenance hash mismatch: {path_key}")

    sources = ledger.get("sources")
    require(isinstance(sources, list) and sources, "sources are missing")
    source_ids = [source.get("id") for source in sources]
    require(len(source_ids) == len(set(source_ids)), "duplicate source ID")
    for source in sources:
        parsed = urlparse(source.get("url", ""))
        require(parsed.scheme == "https" and bool(parsed.netloc), f"invalid source URL: {source.get('id')}")
        require(source.get("authority") in {
            "secondary_jma_display",
            "official_jtwc",
            "research_guidance_archive",
        }, f"unknown source authority: {source.get('id')}")

    data = ledger.get("data")
    require(isinstance(data, list) and data, "data is missing")
    data_ids = [item.get("id") for item in data]
    require(len(data_ids) == len(set(data_ids)), "duplicate data ID")
    for item in data:
        require(isinstance(item.get("text"), str) and item["text"], f"empty data text: {item.get('id')}")
        selected_sources = item.get("source_ids")
        require(isinstance(selected_sources, list) and selected_sources, f"source_ids missing: {item.get('id')}")
        require(set(selected_sources) <= set(source_ids), f"unknown source ID: {item.get('id')}")
        observed = parse_time(item.get("observed_at"), f"{item.get('id')}.observed_at")
        valid_until = parse_time(item.get("valid_until"), f"{item.get('id')}.valid_until")
        require(observed <= cutoff <= valid_until, f"cutoff outside evidence window: {item.get('id')}")

    claims = ledger.get("claims")
    require(isinstance(claims, list) and claims, "claims are missing")
    claim_codes = [claim.get("code") for claim in claims]
    require(len(claim_codes) == len(set(claim_codes)), "duplicate claim code")
    by_code = {claim["code"]: claim for claim in claims}
    for claim in claims:
        require(isinstance(claim.get("label"), str) and claim["label"], f"empty claim label: {claim.get('code')}")
        supported = claim.get("supported_by")
        require(isinstance(supported, list) and supported, f"unsupported claim: {claim.get('code')}")
        require(len(supported) == len(set(supported)), f"duplicate supported_by: {claim.get('code')}")
        require(set(supported) <= set(data_ids), f"unknown supported_by: {claim.get('code')}")
        terms = claim.get("required_terms")
        require(isinstance(terms, list) and terms, f"required_terms missing: {claim.get('code')}")
        require(all(term in claim["label"] for term in terms), f"required term outside label: {claim.get('code')}")
        contradicts = claim.get("contradicts")
        require(isinstance(contradicts, list), f"contradicts must be an array: {claim.get('code')}")
        require(set(contradicts) <= set(claim_codes) - {claim["code"]}, f"unknown/self contradiction: {claim.get('code')}")
    for code, claim in by_code.items():
        for other in claim["contradicts"]:
            require(code in by_code[other]["contradicts"], f"asymmetric contradiction: {code}/{other}")

    personas = ledger.get("personas")
    require(isinstance(personas, list) and len(personas) == 4, "exactly four personas are required")
    persona_names = [persona.get("name") for persona in personas]
    require(len(persona_names) == len(set(persona_names)), "duplicate persona")
    require(all(persona.get("objective") for persona in personas), "persona objective is missing")
    preferences = ledger.get("role_preferences")
    require(isinstance(preferences, dict) and set(preferences) == set(persona_names), "role_preferences mismatch")
    for persona, codes in preferences.items():
        require(isinstance(codes, list) and len(codes) >= 2, f"too few preferences: {persona}")
        require(len(codes) == len(set(codes)), f"duplicate role preference: {persona}")
        require(set(codes) <= set(claim_codes), f"unknown role preference: {persona}")

    readiness_sha = None
    if args.readiness:
        readiness = json.loads(args.readiness.read_text())
        readiness_sha = sha256(args.readiness)
        require(readiness.get("schema_version") == 1, "readiness schema mismatch")
        require(readiness.get("ledger_sha256") == sha256(args.ledger), "readiness ledger hash mismatch")
        require(readiness.get("physical_iphone_run_completed") is False, "readiness overclaims iPhone completion")
        require(readiness.get("promotion_allowed") is False, "readiness must not promote")

        choices = readiness.get("structure", {}).get("choices")
        require(isinstance(choices, list) and choices, "readiness choices missing")
        valid_choices = [choice for choice in choices if choice.get("valid") is True]
        require(len(valid_choices) == 4, "readiness must end with four valid choices")
        require(len({choice["persona"] for choice in valid_choices}) == 4, "valid choice persona mismatch")
        for choice in valid_choices:
            code = choice.get("claim")
            selected = choice.get("data_ids")
            require(code in by_code, f"readiness unknown claim: {code}")
            require(set(selected or []) <= set(by_code[code]["supported_by"]), f"readiness unsupported data: {code}")
            require(1 <= choice.get("confidence", 0) <= 100, f"readiness confidence invalid: {code}")
        invalid_choices = [choice for choice in choices if choice.get("valid") is False]
        require(
            len(invalid_choices) == readiness["structure"]["repairs"] <= 1,
            "repair audit mismatch",
        )
        require(all(choice.get("error") for choice in invalid_choices), "invalid choice lacks error")

        body_outputs = readiness.get("body", {}).get("outputs")
        require(isinstance(body_outputs, list) and len(body_outputs) == len(claims), "body readiness count mismatch")
        require({row.get("claim") for row in body_outputs} == set(claim_codes), "body readiness claim set mismatch")
        sanitized_count = 0
        for row in body_outputs:
            claim = by_code[row["claim"]]
            normalized, error = cod_model.normalize_renderer_body(row.get("raw_body"))
            require(normalized is not None, f"invalid readiness body: {row['claim']} {error}")
            sanitized = False
            if not cod_model.body_is_polite_sentence(normalized):
                candidate = cod_model.sanitize_body_politeness(normalized, claim["label"])
                require(candidate is not None, f"unsanitizable readiness body: {row['claim']}")
                normalized = candidate
                sanitized = True
            require(sanitized is row.get("sanitized"), f"body sanitizer flag mismatch: {row['claim']}")
            require(cod_model.body_matches_claim(normalized, claim["label"]), f"body/claim mismatch: {row['claim']}")
            require(cod_model.body_is_neutral(normalized), f"body move leak: {row['claim']}")
            require(
                set(re.findall(r"\d+(?:\.\d+)?", normalized))
                <= set(re.findall(r"\d+(?:\.\d+)?", claim["label"])),
                f"body invented a number: {row['claim']}",
            )
            sanitized_count += int(sanitized)
        require(sanitized_count == 3, "unexpected body sanitization count")
        require(readiness["body"].get("fallbacks") == 0, "readiness body fallback used")

    print(json.dumps({
        "status": "valid",
        "scenario_id": ledger["scenario_id"],
        "cutoff": cutoff.isoformat(),
        "sources": len(sources),
        "data": len(data),
        "claims": len(claims),
        "personas": len(personas),
        "sha256": sha256(args.ledger),
        "readiness_sha256": readiness_sha,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
