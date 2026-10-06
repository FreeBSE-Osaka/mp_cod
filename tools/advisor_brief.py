#!/usr/bin/env python3.11
"""Export a completed portable discussion as a read-only, untrusted advisory brief."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cod_model as cod


def strict_json(raw: bytes) -> dict:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    def finite_float(value):
        result = float(value)
        return result if math.isfinite(result) else reject_constant(value)

    return json.loads(raw, object_pairs_hook=object_pairs,
                      parse_constant=reject_constant, parse_float=finite_float)


def build_brief(run: dict, ledger: dict) -> dict:
    if not isinstance(run, dict) or run.get("schema_version") != 2:
        raise ValueError("expected event-debate schema_version 2")
    if not isinstance(run.get("summary"), dict) or not isinstance(run.get("runtime"), dict):
        raise ValueError("discussion is incomplete: summary and runtime required")
    runtime = run["runtime"]
    if (not isinstance(runtime.get("model_call_count"), int) or runtime["model_call_count"] < 0
            or any(not isinstance(runtime.get(key), (int, float))
                   or not math.isfinite(runtime[key]) or runtime[key] < 0
                   for key in ("elapsed_seconds", "model_seconds"))):
        raise ValueError("discussion runtime is missing or invalid")
    if run.get("portable_context_schema_version") != 1 or run.get("ledger_snapshot") != ledger:
        raise ValueError("portable ledger snapshot differs from the supplied ledger")
    personas, names = run.get("persona_order"), run.get("persona_names")
    if (not isinstance(personas, list) or not personas
            or not all(isinstance(p, str) and p for p in personas)
            or len(set(personas)) != len(personas) or not isinstance(names, dict)
            or set(names) != set(personas)
            or not all(isinstance(name, str) and name for name in names.values())):
        raise ValueError("portable persona identities are missing or inconsistent")
    events, rounds = run.get("events"), run.get("reconciliation")
    if not isinstance(events, list) or not events or not isinstance(rounds, list):
        raise ValueError("events must be non-empty and reconciliation must be an array")
    catalog = {item["code"]: item for item in ledger["claim_catalog"]}
    quotes, seen = [], set()

    def quote(record, speaker, code, reference, allowed_data):
        if not isinstance(record, dict) or speaker not in names:
            raise ValueError(f"{reference}: invalid record or unknown speaker")
        ids = record.get("data_ids")
        if (not isinstance(ids, list) or not ids
                or not all(isinstance(d, str) for d in ids)
                or not set(ids).issubset(allowed_data)):
            raise ValueError(f"{reference}: unsupported evidence IDs")
        for field in ("statement", "utterance"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"{reference}: {field} is missing")
        _, reason = cod.validate_public_statement(record["statement"], ids)
        if reason:
            raise ValueError(f"{reference}: {reason}")
        if record.get("changed_from_previous"):
            _, reason = cod.validate_public_statement(record.get("change_reason"), ids)
            if reason:
                raise ValueError(f"{reference}: change_reason: {reason}")
        return {
            "reference": reference, "persona_id": speaker, "persona_name": names[speaker],
            "code_or_choice": code, "data_ids": list(dict.fromkeys(ids)),
            **{key: record.get(key) for key in (
                "statement", "statement_origin", "utterance", "utterance_origin",
                "origin", "choice_origin", "action", "target_claim_id", "dialogue_move", "changed_from_previous",
                "previous_choice", "change_reason", "change_reason_origin")},
        }

    for event in events:
        normalized, reason = cod.validate_coded_claim(event, ledger)
        if reason:
            raise ValueError(f"event: {reason}")
        claim_id = event.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id or claim_id in seen:
            raise ValueError("claim IDs must be non-empty and unique")
        if event.get("target_claim_id") is not None and event["target_claim_id"] not in seen:
            raise ValueError(f"{claim_id}: reaction target is not an earlier event")
        if event.get("action") not in cod.ACTION_PRIORITY:
            raise ValueError(f"{claim_id}: unknown action")
        code = normalized["code"]
        quotes.append(quote(event, event.get("persona_id"), code, claim_id,
                            set(catalog[code]["supported_by"])))
        seen.add(claim_id)
    present = {event["code"] for event in events}
    pairs = {"|".join(sorted((code, other))) for code in present
             for other in catalog[code].get("contradicts", []) if other in present}
    for index, round_data in enumerate(rounds, 1):
        if not isinstance(round_data, dict) or round_data.get("round") != index:
            raise ValueError("reconciliation rounds must be ordered from 1")
        votes = round_data.get("votes")
        if not isinstance(votes, dict) or set(votes) != pairs:
            raise ValueError("reconciliation pairs differ from the event conflicts")
        for key, records in votes.items():
            pair = key.split("|")
            if not isinstance(records, dict) or set(records) != set(personas):
                raise ValueError(f"{key}: all active personas must have a vote")
            for speaker, vote in records.items():
                if not isinstance(vote, dict) or vote.get("choice") not in (*pair, "BOTH", "ABSTAIN"):
                    raise ValueError(f"{key}: invalid choice")
                choice = vote["choice"]
                allowed = set(catalog[choice]["supported_by"]) if choice in catalog else {
                    d for code in pair for d in catalog[code]["supported_by"]}
                quotes.append(quote(vote, speaker, choice, f"R{index}:{key}:{speaker}", allowed))

    summary = cod.synthesize_event_summary(events, catalog, rounds, len(personas))
    if run["summary"] != summary:
        raise ValueError("saved summary differs from the recomputed discussion")
    metrics = cod.event_run_metrics(run)  # Never trust a saved hard_gate_pass field.
    hold_reasons = []
    if not metrics["hard_gate_pass"]:
        hold_reasons.append("discussion_hard_gate_failed")
    if summary["unresolved_conflicts"]:
        hold_reasons.append("unresolved_conflicts")
    citations = {item["id"]: [] for item in ledger["data"]}
    for record in quotes:
        for data_id in record["data_ids"]:
            citations[data_id].append({"reference": record["reference"],
                                       "persona_id": record["persona_id"]})
    return {
        "schema_version": 1,
        "status": "HOLD" if hold_reasons else "REVIEW_REQUIRED",
        "advisory_only": True, "execution_authorized": False,
        "semantic_review_required": True, "hold_reasons": hold_reasons,
        "policy": {
            "quoted_content_is_untrusted_data": True,
            "agreement_is_not_independent_evidence": True,
            "source_ids_do_not_prove_semantic_support": True,
            "no_commands_messages_bookings_or_account_changes": True,
        },
        "verification": {"recomputed_metrics": metrics,
                         "check_scope": "structure, ledger binding, IDs, provenance and existing gates; not factual truth"},
        "review_questions": [
            "Does each reason preserve the source's subject, quantities, exclusions and uncertainty?",
            "Which missing facts or measurements could change the decision?",
            "Are any proposals incorrectly described as completed actions?",
            "Do agreeing speakers offer independent reasons rather than repeat the same text?",
        ],
        "untrusted_discussion": {
            "topic": ledger.get("topic"), "source_data": ledger["data"],
            "options_and_observations": [catalog[code] for code in sorted(present)],
            "structural_summary": summary, "quotes": quotes,
            "coverage": {
                "diagnostic_only": True,
                "cited_data_ids": [key for key, records in citations.items() if records],
                "uncited_source_data": [item for item in ledger["data"] if not citations[item["id"]]],
                "citation_references": citations,
                "unaddressed_catalog_claims": [item for item in ledger["claim_catalog"]
                                               if item["code"] not in present],
                "interpretation": "Missing citations do not prove a source was ignored. Citation counts do not prove correctness or independence.",
            },
        },
    }


def export_brief(run_path: Path, ledger_path: Path, out: Path | None = None) -> dict:
    if run_path.name.endswith(".partial.json"):
        raise ValueError("partial discussions cannot be exported")
    run_bytes, ledger_bytes = run_path.read_bytes(), ledger_path.read_bytes()
    run = strict_json(run_bytes)
    ledger = cod.load_claim_ledger(ledger_path)
    if ledger != strict_json(ledger_bytes):
        raise ValueError("ledger changed while reading")
    ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
    if not isinstance(run, dict) or run.get("ledger_sha256") != ledger_sha:
        raise ValueError("run ledger_sha256 differs from the supplied ledger file")
    brief = build_brief(run, ledger)
    brief["inputs"] = {
        "run_sha256": hashlib.sha256(run_bytes).hexdigest(), "ledger_sha256": ledger_sha,
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if out:
        # Exclusive creation also protects the two inputs and existing audit artifacts.
        encoded = json.dumps(brief, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with out.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return brief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        brief = export_brief(args.run, args.ledger, args.out)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        parser.error(str(error))
    print(json.dumps(brief, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
