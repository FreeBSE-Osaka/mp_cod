#!/usr/bin/env python3
"""Multiple Personality CoD: independent expert roles, evidence, and public debate."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import itertools
import json
import math
import random
import re
import sys
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path


DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_API_URL = "http://127.0.0.1:11434/api/chat"
PROFILE_PATH = Path(__file__).with_name("personas.json")
STANCES = ["主案", "対案", "条件付き", "保留"]


def short_string(limit: int = 160) -> dict:
    return {"type": "string", "maxLength": limit}


BLIND_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "stance": {"type": "string", "enum": STANCES},
        "thesis": short_string(),
        "recommendation": short_string(),
        "reasons": {"type": "array", "items": short_string(120), "minItems": 1, "maxItems": 2},
        "assumptions": {"type": "array", "items": short_string(120), "maxItems": 2},
        "risks": {"type": "array", "items": short_string(120), "maxItems": 2},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["stance", "thesis", "recommendation", "reasons", "assumptions", "risks", "confidence"],
}


MODERATOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": short_string(180),
        "final_answer": short_string(160),
        "decisive_evidence": {"type": "array", "items": short_string(120), "maxItems": 2},
        "rejected_or_weaker": {"type": "array", "items": short_string(120), "maxItems": 2},
        "unresolved": {"type": "array", "items": short_string(120), "maxItems": 2},
        "minority_report": short_string(160),
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": [
        "decision",
        "final_answer",
        "decisive_evidence",
        "rejected_or_weaker",
        "unresolved",
        "minority_report",
        "confidence",
    ],
}


VERIFIER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
        "checked_claims": {"type": "array", "items": short_string(120), "maxItems": 2},
        "issues": {"type": "array", "items": short_string(120), "maxItems": 2},
        "final_expression": short_string(80),
        "corrected_final_answer": short_string(160),
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": [
        "verdict",
        "checked_claims",
        "issues",
        "final_expression",
        "corrected_final_answer",
        "confidence",
    ],
}


def review_schema(other_ids: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_persona": {"type": "string", "enum": other_ids},
            "challenge": short_string(),
            "response": short_string(),
            "revised_stance": {"type": "string", "enum": STANCES},
            "revised_recommendation": short_string(),
            "changed": {"type": "boolean"},
            "remaining_risk": short_string(),
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": [
            "target_persona",
            "challenge",
            "response",
            "revised_stance",
            "revised_recommendation",
            "changed",
            "remaining_risk",
            "confidence",
        ],
    }


def load_domains(path: Path = PROFILE_PATH) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    domains = payload.get("domains")
    if payload.get("schema_version") != 1 or not isinstance(domains, dict):
        raise ValueError(f"Unsupported persona file: {path}")
    for domain, config in domains.items():
        personas = config.get("personas")
        if not isinstance(personas, list) or len(personas) < 2:
            raise ValueError(f"{domain}: at least two personas are required")
        ids = [persona.get("id") for persona in personas]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{domain}: persona ids must be non-empty and unique")
        for persona in personas:
            required = ("name", "worldview", "objective", "tests", "avoid")
            if any(not persona.get(key) for key in required):
                raise ValueError(f"{domain}/{persona['id']}: missing persona field")
    return domains


def persona_system(persona: dict, phase: str) -> str:
    return f"""あなたは同一基盤モデル内の独立した専門家人格「{persona['name']}」です。
世界観: {persona['worldview']}
目的: {persona['objective']}
検査: {' / '.join(persona['tests'])}
禁止: {' / '.join(persona['avoid'])} / 迎合 / 多数決 / 未知の固有名の創作
フェーズ: {phase}
公開可能な根拠だけをJSONで返し、内部思考は出さないでください。
必須キー: stance, thesis, recommendation, reasons, assumptions, risks, confidence
stanceは主案・対案・条件付き・保留。各文字列160字以内、配列最大2件。
証拠不足は保留、反証時は修正。confidenceは未検証30〜70、独立検証済みのみ100。
"""


def stable_seed(base: int, persona_id: str, phase: str) -> int:
    digest = hashlib.sha256(f"{persona_id}:{phase}".encode()).hexdigest()
    return base + int(digest[:8], 16) % 1_000_000


def ask_ollama(
    *,
    model: str,
    system: str,
    user: str,
    schema: dict,
    api_url: str,
    timeout: int,
    num_predict: int,
    temperature: float,
    seed: int,
) -> tuple[dict, dict]:
    last_error: Exception | None = None
    for attempt in range(2):
        repair = "\n前回は形式不正でした。指定JSONだけを完全に出力してください。" if attempt else ""
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user + repair},
            ],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
                "num_ctx": 8192,
            },
            "keep_alive": "10m",
        }
        request = urllib.request.Request(
            api_url,
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            content = payload.get("message", {}).get("content", "")
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("model output is not a JSON object")
            eval_seconds = payload.get("eval_duration", 0) / 1_000_000_000
            meta = {
                "attempts": attempt + 1,
                "eval_count": payload.get("eval_count", 0),
                "tokens_per_second": round(payload.get("eval_count", 0) / eval_seconds, 2) if eval_seconds else 0,
                "total_seconds": round(payload.get("total_duration", 0) / 1_000_000_000, 3),
            }
            return result, meta
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"Ollama call failed after retry: {last_error}")


def call_record(phase: str, persona_id: str, system: str, user: str, response: dict, meta: dict) -> dict:
    return {
        "phase": phase,
        "persona_id": persona_id,
        "system": system,
        "user": user,
        "assistant": json.dumps(response, ensure_ascii=False),
        "response": response,
        "runtime": meta,
    }


def normalized_ngrams(text: str, size: int = 3) -> set[str]:
    normalized = re.sub(r"[^\w]+", "", text.casefold())
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def similarity(left: str, right: str) -> float:
    a, b = normalized_ngrams(left), normalized_ngrams(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def public_text(response: dict) -> str:
    return " ".join(
        str(value)
        for value in [response.get("thesis"), response.get("recommendation"), *(response.get("reasons") or [])]
        if value
    )


def debate_metrics(blind: dict[str, dict], reviews: dict[str, dict]) -> dict:
    pairs = [
        {"personas": [left_id, right_id], "similarity": round(similarity(public_text(left), public_text(right)), 3)}
        for (left_id, left), (right_id, right) in itertools.combinations(blind.items(), 2)
    ]
    similarities = [pair["similarity"] for pair in pairs]
    stances = {response.get("stance") for response in blind.values()}
    mean_similarity = sum(similarities) / len(similarities) if similarities else 1.0
    review_pairs = [
        similarity(str(left.get("revised_recommendation", "")), str(right.get("revised_recommendation", "")))
        for left, right in itertools.combinations(reviews.values(), 2)
    ]
    review_mean = sum(review_pairs) / len(review_pairs) if review_pairs else 1.0
    review_stances = {response.get("revised_stance") for response in reviews.values()}
    return {
        "blind_personas": len(blind),
        "blind_stance_count": len(stances),
        "mean_pairwise_similarity": round(mean_similarity, 3),
        "max_pairwise_similarity": max(similarities, default=1.0),
        "near_duplicate_pairs": sum(value >= 0.8 for value in similarities),
        "zero_confidence_personas": sum(response.get("confidence") == 0 for response in blind.values()),
        "review_change_rate": round(sum(bool(item.get("changed")) for item in reviews.values()) / len(reviews), 3)
        if reviews
        else 0,
        "review_stance_count": len(review_stances),
        "review_mean_pairwise_similarity": round(review_mean, 3),
        "convergence_delta": round(review_mean - mean_similarity, 3),
        "premature_consensus_proxy": len(blind) > 1 and len(stances) == 1 and mean_similarity >= 0.55,
        "post_review_consensus_proxy": len(reviews) > 1 and len(review_stances) == 1 and review_mean >= 0.55,
        "pairs": pairs,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_partial(path: Path, run: dict) -> None:
    run["updated_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(path, run)


def calculate(expression: str) -> str:
    operators = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
    }

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("四則演算と数値だけを使用できます")

    if len(expression) > 80:
        raise ValueError("式が長すぎます")
    value = evaluate(ast.parse(expression, mode="eval"))
    if abs(value.numerator) > 10**18 or value.denominator > 10**18:
        raise ValueError("計算結果が大きすぎます")
    fraction = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return f"{fraction} = {float(value):.12g}"


def effective_answer(moderator: dict, verifier: dict | None) -> str:
    if verifier and verifier.get("calculator_result"):
        return str(verifier["calculator_result"])
    if verifier and verifier.get("verdict") == "fail" and verifier.get("corrected_final_answer"):
        return str(verifier["corrected_final_answer"])
    if verifier and verifier.get("verdict") == "uncertain":
        return f"監査未確定: {moderator['final_answer']}"
    if not verifier:
        return f"未監査: {moderator['final_answer']}"
    return str(moderator["final_answer"])


def run_debate(args: argparse.Namespace) -> int:
    domains = load_domains()
    if args.domain not in domains:
        raise ValueError(f"Unknown domain {args.domain!r}; choose from {', '.join(domains)}")
    domain = domains[args.domain]
    personas = domain["personas"]
    now = dt.datetime.now().astimezone()
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")
    partial_path = Path(args.out) / f"{run_id}_{args.domain}.partial.json"
    final_path = Path(args.out) / f"{run_id}_{args.domain}.json"
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": now.isoformat(timespec="seconds"),
        "topic": args.topic,
        "domain": args.domain,
        "domain_description": domain["description"],
        "model": args.model,
        "personas": personas,
        "calls": [],
        "blind": {},
        "reviews": {},
        "errors": [],
        "training_review": {"status": "unreviewed"},
    }
    save_partial(partial_path, run)

    for persona in personas:
        print(f"[盲検] {persona['name']} -> {args.model}", file=sys.stderr, flush=True)
        system = persona_system(persona, "盲検初期見解。他人格の発言は見えていない。")
        user = f"テーマ:\n{args.topic}\n\n自分の専門観点から独立した初期見解を出してください。"
        try:
            response, meta = ask_ollama(
                model=args.model,
                system=system,
                user=user,
                schema=BLIND_SCHEMA,
                api_url=args.api_url,
                timeout=args.timeout,
                num_predict=args.num_predict,
                temperature=float(persona.get("temperature", 0.35)),
                seed=stable_seed(args.seed, persona["id"], "blind"),
            )
            run["blind"][persona["id"]] = response
            run["calls"].append(call_record("blind", persona["id"], system, user, response, meta))
            print(f"\n### {persona['name']} / {response['stance']} ({response['confidence']}%)")
            print(response["thesis"])
            print(f"提案: {response['recommendation']}")
        except RuntimeError as exc:
            run["errors"].append({"phase": "blind", "persona_id": persona["id"], "error": str(exc)})
            print(f"[失敗] {persona['name']}: {exc}", file=sys.stderr, flush=True)
        save_partial(partial_path, run)

    if len(run["blind"]) < 2:
        print(f"討論に必要な初期見解が不足しました。部分ログ: {partial_path}", file=sys.stderr)
        return 1

    blind_context = json.dumps(run["blind"], ensure_ascii=False, indent=2)
    for persona in personas:
        if persona["id"] not in run["blind"]:
            continue
        other_ids = [item["id"] for item in personas if item["id"] != persona["id"] and item["id"] in run["blind"]]
        print(f"[反論] {persona['name']} -> {args.model}", file=sys.stderr, flush=True)
        system = persona_system(persona, "相互反論。全員の盲検初期見解だけを同時に受け取った。")
        user = f"""テーマ:
{args.topic}

盲検初期見解:
{blind_context}

他人格から最も強く反対すべき一人を選び、具体的に反証してください。
その反証を踏まえ、自分の初期見解を維持するか修正するかも明示してください。
全員が同意していても、共通の隠れた前提を一つ攻撃してください。"""
        try:
            response, meta = ask_ollama(
                model=args.model,
                system=system,
                user=user,
                schema=review_schema(other_ids),
                api_url=args.api_url,
                timeout=args.timeout,
                num_predict=args.num_predict,
                temperature=float(persona.get("temperature", 0.35)),
                seed=stable_seed(args.seed, persona["id"], "review"),
            )
            run["reviews"][persona["id"]] = response
            run["calls"].append(call_record("review", persona["id"], system, user, response, meta))
            target_name = next(item["name"] for item in personas if item["id"] == response["target_persona"])
            print(f"\n### {persona['name']} → {target_name}")
            print(response["challenge"])
            print(f"修正後: {response['revised_recommendation']}")
        except RuntimeError as exc:
            run["errors"].append({"phase": "review", "persona_id": persona["id"], "error": str(exc)})
            print(f"[失敗] {persona['name']}: {exc}", file=sys.stderr, flush=True)
        save_partial(partial_path, run)

    print(f"[司会] 証拠統合 -> {args.model}", file=sys.stderr, flush=True)
    moderator_system = """あなたはCoD司会者です。参加者と同じ基盤モデルですが、どの学派にも属しません。
多数決、肩書、文章の流暢さではなく、検証可能な証拠と明示された前提で採決してください。
合意を強制せず、少数意見と未解決点を必ず残してください。隠れた思考過程は出さず、指定JSONだけを返してください。"""
    moderator_user = f"""テーマ:
{args.topic}

盲検初期見解:
{json.dumps(run['blind'], ensure_ascii=False, indent=2)}

相互反論:
{json.dumps(run['reviews'], ensure_ascii=False, indent=2)}

実行可能な暫定結論を出してください。証拠不足なら、その条件付き結論と次の検証をdecisionに含めてください。
final_answerだけを読んでもテーマの要求へ直接回答できるようにし、要求された数値・分数・形式を省略しないでください。"""
    try:
        moderator, meta = ask_ollama(
            model=args.model,
            system=moderator_system,
            user=moderator_user,
            schema=MODERATOR_SCHEMA,
            api_url=args.api_url,
            timeout=args.timeout,
            num_predict=max(args.num_predict, 640),
            temperature=0.2,
            seed=stable_seed(args.seed, "moderator", "final"),
        )
        run["moderator"] = moderator
        run["calls"].append(call_record("moderator", "moderator", moderator_system, moderator_user, moderator, meta))
    except RuntimeError as exc:
        run["errors"].append({"phase": "moderator", "persona_id": "moderator", "error": str(exc)})
        save_partial(partial_path, run)
        print(f"司会に失敗しました。部分ログ: {partial_path}\n{exc}", file=sys.stderr)
        return 1

    print(f"[監査] 司会結論を独立検算 -> {args.model}", file=sys.stderr, flush=True)
    verifier_system = """あなたはCoDの独立最終監査者です。司会や多数派へ従ってはいけません。
最終回答、決定的証拠、計算結果の内部整合性をゼロから検査してください。
数値問題は式を再計算し、IT等の事実は未確認の固有名や性能断定をfailまたはuncertainにしてください。
四則演算で最終値を検証できる場合はfinal_expressionへ式だけ（例: 16/50）を書いてください。数値問題でなければ空文字にしてください。
隠れた思考過程は出さず、検査した主張、問題点、必要なら訂正した最終回答だけを指定JSONで返してください。"""
    verifier_user = f"""テーマ:
{args.topic}

司会出力:
{json.dumps(moderator, ensure_ascii=False, indent=2)}

相互反論:
{json.dumps(run['reviews'], ensure_ascii=False, indent=2)}

司会のfinal_answerが証拠と一致するか独立に検査してください。"""
    verifier = None
    try:
        verifier, meta = ask_ollama(
            model=args.model,
            system=verifier_system,
            user=verifier_user,
            schema=VERIFIER_SCHEMA,
            api_url=args.api_url,
            timeout=args.timeout,
            num_predict=max(args.num_predict, 480),
            temperature=0.1,
            seed=stable_seed(args.seed, "verifier", "final"),
        )
        expression = str(verifier.get("final_expression") or "").strip()
        if expression:
            try:
                verifier["calculator_result"] = calculate(expression)
            except (SyntaxError, ValueError, ZeroDivisionError) as exc:
                verifier["issues"].append(f"計算器が式を拒否: {exc}")
                verifier["verdict"] = "uncertain"
        run["verifier"] = verifier
        run["calls"].append(call_record("verifier", "verifier", verifier_system, verifier_user, verifier, meta))
    except RuntimeError as exc:
        run["errors"].append({"phase": "verifier", "persona_id": "verifier", "error": str(exc)})

    run["effective_final_answer"] = effective_answer(moderator, verifier)
    run["metrics"] = debate_metrics(run["blind"], run["reviews"])
    run["metrics"]["moderator_confidence"] = moderator["confidence"]
    run["metrics"]["verifier_verdict"] = verifier.get("verdict") if verifier else "error"
    run["metrics"]["verifier_confidence"] = verifier.get("confidence") if verifier else 0
    save_partial(partial_path, run)
    partial_path.replace(final_path)
    print("\n## Codex司会まとめ")
    print(moderator["decision"])
    print(f"司会回答: {moderator['final_answer']}")
    if verifier:
        print(f"独立監査: {verifier['verdict']} ({verifier['confidence']}%)")
        if verifier["issues"]:
            print(f"監査指摘: {' / '.join(verifier['issues'])}")
    print(f"有効な最終回答: {run['effective_final_answer']}")
    print(f"少数意見: {moderator['minority_report']}")
    print(f"未解決: {' / '.join(moderator['unresolved']) or 'なし'}")
    print(f"確信度: {moderator['confidence']}%")
    print(f"\n同文崩壊proxy: {run['metrics']['premature_consensus_proxy']}")
    print(f"平均類似度: {run['metrics']['mean_pairwise_similarity']}")
    print(f"反論後の収束差: {run['metrics']['convergence_delta']}")
    print(f"保存: {final_path}")
    return 0


def split_examples(examples: list[dict]) -> dict[str, list[dict]]:
    # ponytail: chronological 80/10/10; add topic-aware grouping only after measured leakage.
    count = len(examples)
    holdout = max(1, round(count * 0.1)) if count >= 10 else 0
    test_start = count - holdout
    valid_start = test_start - holdout
    return {
        "train": examples[:valid_start],
        "valid": examples[valid_start:test_start],
        "test": examples[test_start:],
    }


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def make_auditor_curriculum(args: argparse.Namespace) -> int:
    if args.count < 30:
        raise ValueError("curriculumは最低30件にしてください")
    rng = random.Random(args.seed)
    persona = next(item for item in load_domains()["general"]["personas"] if item["id"] == "empirical_auditor")
    system = persona_system(persona, "決定論的な有限集合監査。式と公開根拠だけを回答する。")
    examples: list[dict] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    # ponytail: first Weight proves one measurable skill; add other audit families after holdout gains plateau.
    while len(examples) < args.count:
        upper = rng.randint(40, 500)
        condition = rng.randint(2, 10)
        target = rng.randint(2, 12)
        denominator = upper // condition
        numerator = upper // math.lcm(condition, target)
        answer = Fraction(numerator, denominator)
        if rng.random() < 0.5:
            hypothesis = answer
        else:
            candidates = [
                Fraction(max(0, numerator - 1), denominator),
                Fraction(min(denominator, numerator + 1), denominator),
                Fraction(1, target),
                Fraction(upper // target, upper),
            ]
            candidates = [candidate for candidate in candidates if candidate != answer]
            hypothesis = rng.choice(candidates)
        key = (upper, condition, target, hypothesis.numerator, hypothesis.denominator)
        if key in seen:
            continue
        seen.add(key)
        correct = hypothesis == answer
        answer_fraction = fraction_text(answer)
        answer_decimal = f"{float(answer):.12g}"
        hypothesis_fraction = fraction_text(hypothesis)
        hypothesis_decimal = f"{float(hypothesis):.12g}"
        topic = (
            f"1から{upper}までの整数を一様ランダムに1個選び、{condition}の倍数だったと判明した。"
            f"この条件下で{target}の倍数である確率は、ちょうど{hypothesis_fraction}"
            f"（{hypothesis_decimal}）だという仮説を評価する。有限集合を正確に数え、分数と小数を示す。"
        )
        raw_user = f"テーマ:\n{topic}\n\n自分の専門観点から独立した初期見解を出してください。"
        tool_evidence = (
            "決定論的監査ツール結果:\n"
            f"condition_count={denominator}\n"
            f"intersection_count={numerator}\n"
            f"exact_fraction={answer_fraction}\n"
            f"exact_decimal={answer_decimal}\n"
            f"hypothesis_matches={'true' if correct else 'false'}\n"
            "この証拠を改変せず、人格JSONへ整理してください。"
        )
        user = f"{raw_user}\n\n{tool_evidence}"
        response = {
            "stance": "主案" if correct else "対案",
            "thesis": (
                f"仮説は{'正しい' if correct else '誤り'}。条件集合は{condition}の倍数{denominator}個、"
                f"目標との共通部分は{math.lcm(condition, target)}の倍数{numerator}個。"
            ),
            "recommendation": f"条件付き確率は{numerator}/{denominator}={answer_fraction}（{answer_decimal}）。",
            "reasons": [
                f"分母は⌊{upper}/{condition}⌋={denominator}。",
                f"分子は⌊{upper}/lcm({condition},{target})⌋={numerator}。",
            ],
            "assumptions": [f"1から{upper}は一様分布。", f"条件は{condition}の倍数であること。"],
            "risks": [f"{target}の倍数全体と条件集合との共通部分を混同しない。"],
            "confidence": 100,
        }
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
                ],
                "benchmark": {
                    "system": system,
                    "user": raw_user,
                    "expected_fraction": answer_fraction,
                    "expected_decimal": answer_decimal,
                    "hypothesis_correct": correct,
                },
                "evidence_benchmark": {
                    "system": system,
                    "user": user,
                    "expected_fraction": answer_fraction,
                    "expected_decimal": answer_decimal,
                    "hypothesis_correct": correct,
                },
            }
        )

    rng.shuffle(examples)
    test_count = max(1, round(args.count * 0.1))
    valid_count = max(1, round(args.count * 0.1))
    train_count = args.count - valid_count - test_count
    splits = {
        "train": examples[:train_count],
        "valid": examples[train_count : train_count + valid_count],
        "test": examples[train_count + valid_count :],
    }
    out = Path(args.out)
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", [{"messages": row["messages"]} for row in rows])
    write_jsonl(out / "benchmark.jsonl", [row["benchmark"] for row in splits["test"]])
    write_jsonl(out / "evidence_benchmark.jsonl", [row["evidence_benchmark"] for row in splits["test"]])
    files = [out / f"{name}.jsonl" for name in ("train", "valid", "test", "benchmark", "evidence_benchmark")]
    manifest = {
        "schema_version": 1,
        "generator": "finite_conditional_multiples_with_tool_evidence_v2",
        "seed": args.seed,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "scope": "実証監査者の証拠解釈Weight用。raw benchmarkで算術保持も測定。",
    }
    write_json(out / "manifest.json", manifest)
    print(f"curriculum: {out} {manifest['counts']}")
    return 0


def parse_json_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def score_auditor_output(text: str, expected_fraction: str, hypothesis_correct: bool) -> dict:
    parsed = parse_json_object(text)
    searchable = text if parsed is None else " ".join(str(value) for value in parsed.values())
    expected = Fraction(expected_fraction)
    fractions = []
    for numerator, denominator in re.findall(r"(?<!\d)(-?\d+)\s*/\s*(\d+)", searchable):
        if int(denominator):
            fractions.append(Fraction(int(numerator), int(denominator)))
    contract_valid = bool(parsed) and (
        set(BLIND_SCHEMA["required"]).issubset(parsed)
        and parsed.get("stance") in STANCES
        and isinstance(parsed.get("thesis"), str)
        and isinstance(parsed.get("recommendation"), str)
        and all(
            isinstance(parsed.get(key), list)
            and all(isinstance(value, str) for value in parsed[key])
            and len(parsed[key]) <= 2
            for key in ("reasons", "assumptions", "risks")
        )
        and isinstance(parsed.get("confidence"), int)
    )
    verdict_text = text if parsed is None else f"{parsed.get('thesis', '')} {parsed.get('recommendation', '')}"
    correct_phrase = any(
        phrase in verdict_text for phrase in ("仮説は正しい", "仮説は真", "仮説を支持", "仮説を採用", "真と判断")
    )
    wrong_phrase = any(
        phrase in verdict_text for phrase in ("仮説は誤り", "仮説は偽", "仮説を棄却", "仮説を否定", "一致しない")
    )
    verdict_correct = (correct_phrase and not wrong_phrase) if hypothesis_correct else (wrong_phrase and not correct_phrase)
    fraction_correct = expected in fractions
    return {
        "json_valid": parsed is not None,
        "contract_valid": contract_valid,
        "fraction_correct": fraction_correct,
        "verdict_correct": verdict_correct,
        "semantic_correct": fraction_correct and verdict_correct,
        "all_correct": contract_valid and fraction_correct and verdict_correct,
    }


def normalize_auditor_contract(value: dict | None) -> dict | None:
    if not value or not set(BLIND_SCHEMA["required"]).issubset(value):
        return None
    if value.get("stance") not in STANCES:
        return None
    if not isinstance(value.get("thesis"), str) or not isinstance(value.get("recommendation"), str):
        return None
    normalized = dict(value)
    for key in ("reasons", "assumptions", "risks"):
        if not isinstance(value.get(key), list) or not all(isinstance(item, str) for item in value[key]):
            return None
        normalized[key] = value[key][:2]
    if not isinstance(value.get("confidence"), int):
        return None
    normalized["confidence"] = max(0, min(100, value["confidence"]))
    return normalized


def evaluate_weight(args: argparse.Namespace) -> int:
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise ValueError("weight-evalはMLX-LM環境のPythonで実行してください") from exc

    rows = [json.loads(line) for line in Path(args.benchmark).read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    results = []
    for index, row in enumerate(rows, 1):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )
        score = score_auditor_output(output, row["expected_fraction"], bool(row["hypothesis_correct"]))
        results.append({"index": index, "expected": row, "output": output, "score": score})
        print(f"[{args.label}] {index}/{len(rows)} all_correct={score['all_correct']}", file=sys.stderr, flush=True)
    count = len(results)
    metrics = {
        "label": args.label,
        "model": args.model,
        "adapter": args.adapter,
        "examples": count,
        "json_valid_rate": round(sum(item["score"]["json_valid"] for item in results) / count, 4),
        "contract_valid_rate": round(sum(item["score"]["contract_valid"] for item in results) / count, 4),
        "fraction_accuracy": round(sum(item["score"]["fraction_correct"] for item in results) / count, 4),
        "verdict_accuracy": round(sum(item["score"]["verdict_correct"] for item in results) / count, 4),
        "semantic_accuracy": round(sum(item["score"]["semantic_correct"] for item in results) / count, 4),
        "all_correct_accuracy": round(sum(item["score"]["all_correct"] for item in results) / count, 4),
    }
    write_json(Path(args.out), {"metrics": metrics, "results": results})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def finite_audit_input(upper: int, condition: int, target: int, hypothesis: Fraction) -> dict:
    if upper < 1 or condition < 1 or target < 1:
        raise ValueError("upper, condition, targetは正の整数です")
    denominator = upper // condition
    if denominator == 0:
        raise ValueError("条件集合が空です")
    numerator = upper // math.lcm(condition, target)
    answer = Fraction(numerator, denominator)
    answer_fraction = fraction_text(answer)
    answer_decimal = f"{float(answer):.12g}"
    hypothesis_fraction = fraction_text(hypothesis)
    hypothesis_decimal = f"{float(hypothesis):.12g}"
    topic = (
        f"1から{upper}までの整数を一様ランダムに1個選び、{condition}の倍数だったと判明した。"
        f"この条件下で{target}の倍数である確率は、ちょうど{hypothesis_fraction}"
        f"（{hypothesis_decimal}）だという仮説を評価する。有限集合を正確に数え、分数と小数を示す。"
    )
    persona = next(item for item in load_domains()["general"]["personas"] if item["id"] == "empirical_auditor")
    system = persona_system(persona, "決定論的ツール証拠を解釈する監査。証拠なしで推測しない。")
    raw_user = f"テーマ:\n{topic}\n\n自分の専門観点から独立した初期見解を出してください。"
    user = (
        f"{raw_user}\n\n決定論的監査ツール結果:\n"
        f"condition_count={denominator}\n"
        f"intersection_count={numerator}\n"
        f"exact_fraction={answer_fraction}\n"
        f"exact_decimal={answer_decimal}\n"
        f"hypothesis_matches={'true' if hypothesis == answer else 'false'}\n"
        "この証拠を改変せず、人格JSONへ整理してください。"
    )
    return {
        "system": system,
        "user": user,
        "expected_fraction": answer_fraction,
        "expected_decimal": answer_decimal,
        "hypothesis_correct": hypothesis == answer,
        "effective_final_answer": f"{answer_fraction} = {answer_decimal}",
    }


def run_weight_audit(args: argparse.Namespace) -> int:
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise ValueError("weight-auditはMLX-LM環境のPythonで実行してください") from exc
    try:
        hypothesis = Fraction(args.hypothesis)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("hypothesisは1/3または0.25形式です") from exc
    case = finite_audit_input(args.upper, args.condition, args.target, hypothesis)
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": case["system"]},
            {"role": "user", "content": case["user"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )
    normalized = normalize_auditor_contract(parse_json_object(output))
    if normalized:
        normalized["recommendation"] = (
            f"{normalized['recommendation'].rstrip('。')}。確定値: {case['effective_final_answer']}。"
        )[:160]
    effective_output = json.dumps(normalized, ensure_ascii=False) if normalized else output
    score = score_auditor_output(effective_output, case["expected_fraction"], case["hypothesis_correct"])
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "tool_evidence_required": True,
        "effective_final_answer": case["effective_final_answer"],
        "weight_output": output,
        "normalized_weight_output": normalized,
        "weight_score": score,
    }
    if args.out:
        write_json(Path(args.out), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


ACTION_LABELS = {
    "object": "異議あり！",
    "agree_extend": "賛同＋補足",
    "new": "新規論点",
}
ACTION_PRIORITY = {"object": 3, "agree_extend": 2, "new": 1}


def load_claim_ledger(path: Path) -> dict:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != 1:
        raise ValueError("claim ledger schema_version must be 1")
    data = ledger.get("data")
    catalog = ledger.get("claim_catalog")
    if not isinstance(data, list) or not isinstance(catalog, list):
        raise ValueError("claim ledger requires data and claim_catalog lists")
    data_ids = [item.get("id") for item in data]
    codes = [item.get("code") for item in catalog]
    if any(not value for value in data_ids + codes) or len(data_ids) != len(set(data_ids)) or len(codes) != len(set(codes)):
        raise ValueError("data ids and claim codes must be non-empty and unique")
    known_data = set(data_ids)
    known_codes = set(codes)
    for claim in catalog:
        supported_by = claim.get("supported_by")
        contradicts = claim.get("contradicts", [])
        if not isinstance(supported_by, list) or not supported_by or not set(supported_by).issubset(known_data):
            raise ValueError(f"{claim['code']}: invalid supported_by")
        if not isinstance(contradicts, list) or not set(contradicts).issubset(known_codes):
            raise ValueError(f"{claim['code']}: invalid contradicts")
    return ledger


def validate_coded_claim(claim: dict, ledger: dict) -> tuple[dict | None, str | None]:
    if not isinstance(claim, dict):
        return None, "claim is not an object"
    code = claim.get("code")
    catalog = {item["code"]: item for item in ledger["claim_catalog"]}
    if code not in catalog:
        return None, f"unknown claim code: {code}"
    data_ids = claim.get("data_ids")
    if not isinstance(data_ids, list) or not data_ids or not all(isinstance(value, str) for value in data_ids):
        return None, "data_ids must be a non-empty string list"
    allowed = set(catalog[code]["supported_by"])
    if not set(data_ids).issubset(allowed):
        return None, f"unsupported data ids for {code}: {sorted(set(data_ids) - allowed)}"
    confidence = claim.get("confidence", 50)
    if isinstance(confidence, str):
        confidence = {"HIGH": 85, "MEDIUM": 60, "LOW": 35}.get(confidence.upper())
    if not isinstance(confidence, int) or not 0 <= confidence <= 100:
        return None, "confidence must be an integer from 0 to 100"
    return {"code": code, "data_ids": list(dict.fromkeys(data_ids)), "confidence": confidence}, None


def claim_reaction(events: list[dict], candidate: dict, catalog: dict[str, dict]) -> tuple[str, str | None]:
    code = candidate["code"]
    for event in reversed(events):
        previous = event["code"]
        if code in catalog[previous].get("contradicts", []) or previous in catalog[code].get("contradicts", []):
            return "object", event["claim_id"]
    for event in reversed(events):
        if code == event["code"]:
            return "agree_extend", event["claim_id"]
    return "new", None


def schedule_claim_events(persona_claims: dict[str, list[dict]], ledger: dict, max_turns: int) -> list[dict]:
    queues = {persona_id: list(claims) for persona_id, claims in persona_claims.items()}
    catalog = {item["code"]: item for item in ledger["claim_catalog"]}
    persona_order = {persona_id: index for index, persona_id in enumerate(queues)}
    speak_count = {persona_id: 0 for persona_id in queues}
    events = []
    last_speaker = None
    while len(events) < max_turns and any(queues.values()):
        candidates = []
        active_speakers = sum(bool(queue) for queue in queues.values())
        for persona_id, queue in queues.items():
            if not queue:
                continue
            claim = queue[0]
            action, target_claim_id = claim_reaction(events, claim, catalog)
            cooldown = -1 if persona_id == last_speaker and active_speakers > 1 else 0
            priority = (
                ACTION_PRIORITY[action],
                int(catalog[claim["code"]].get("priority", 1)),
                claim["confidence"],
                cooldown,
                -speak_count[persona_id],
                -persona_order[persona_id],
            )
            candidates.append((priority, persona_id, claim, action, target_claim_id))
        _, persona_id, claim, action, target_claim_id = max(candidates, key=lambda item: item[0])
        queues[persona_id].pop(0)
        event = {
            "claim_id": f"C{len(events) + 1:02d}",
            "persona_id": persona_id,
            "action": action,
            "action_label": ACTION_LABELS[action],
            "target_claim_id": target_claim_id,
            "code": claim["code"],
            "label": catalog[claim["code"]]["label"],
            "data_ids": claim["data_ids"],
            "confidence": claim["confidence"],
        }
        events.append(event)
        last_speaker = persona_id
        speak_count[persona_id] += 1
    return events


def synthesize_event_summary(
    events: list[dict], catalog: dict[str, dict], reconciliation: list[dict], persona_count: int
) -> dict:
    speakers: dict[str, set[str]] = {}
    for event in events:
        speakers.setdefault(event["code"], set()).add(event["persona_id"])
    present = set(speakers)
    pairs = sorted(
        {
            tuple(sorted((code, other)))
            for code in present
            for other in catalog[code].get("contradicts", [])
            if other in present
        }
    )
    threshold = max(1, (persona_count * 3 + 3) // 4)
    resolved: dict[str, str] = {}
    unresolved = list(pairs)
    if reconciliation:
        final_votes = reconciliation[-1].get("votes", {})
        unresolved = []
        for pair in pairs:
            key = "|".join(pair)
            choices = final_votes.get(key, {})
            counts = {code: sum(choice == code for choice in choices.values()) for code in pair}
            winner = max(counts, key=counts.get)
            if counts[winner] >= threshold:
                resolved[key] = winner
            else:
                unresolved.append(pair)
    excluded = {code for pair in pairs for code in pair}
    for winner in resolved.values():
        excluded.discard(winner)
    consensus = sorted(
        {code for code, owners in speakers.items() if len(owners) >= 2 and code not in excluded}
        | set(resolved.values())
    )
    unopposed = sorted(code for code in present if code not in excluded and code not in consensus)
    return {
        "consensus": consensus,
        "unopposed_supported": unopposed,
        "resolved_conflicts": resolved,
        "unresolved_conflicts": [list(pair) for pair in unresolved],
        "speaker_counts": {code: len(owners) for code, owners in speakers.items()},
    }


def run_event_debate(args: argparse.Namespace) -> int:
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise ValueError("event-debateはMLX-LM環境のPythonで実行してください") from exc
    ledger = load_claim_ledger(Path(args.ledger))
    domains = load_domains()
    personas = domains[args.domain]["personas"]
    catalog = {item["code"]: item for item in ledger["claim_catalog"]}
    data_view = [{"id": item["id"], "text": item["text"]} for item in ledger["data"]]
    catalog_view = [
        {"code": item["code"], "label": item["label"], "supported_by": item["supported_by"]}
        for item in ledger["claim_catalog"]
    ]
    preferences = ledger.get("role_preferences", {})
    print(f"[MLX] {args.model_path} をロード", flush=True)
    model, tokenizer = load(args.model_path)
    sampler = make_sampler(temp=args.temperature, top_p=0.85)

    def ask_json(system: str, user: str, max_tokens: int) -> tuple[str, dict | None]:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False).strip()
        return raw, parse_json_object(raw)

    run = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.model_path,
        "ledger": args.ledger,
        "independent": {},
        "events": [],
        "reconciliation": [],
    }
    persona_claims: dict[str, list[dict]] = {}
    print("\n######## 独立再計算 / 他者の文章は非公開 ########", flush=True)
    for persona in personas:
        available_codes = [code for code in preferences.get(persona["id"], []) if code in catalog]
        available_catalog = [
            {"code": code, "label": catalog[code]["label"], "supported_by": catalog[code]["supported_by"]}
            for code in available_codes
        ]
        system = (
            f"あなたは{persona['name']}。他人格の文章は一切見ていない。"
            "dataとclaim_catalogだけを独立確認し、自由文主張は禁止。指定JSONだけを返す。"
        )
        user = json.dumps(
            {
                "topic": ledger["topic"],
                "data": data_view,
                "claim_catalog": available_catalog,
                "persona_focus": persona["objective"],
                "rule": (
                    "claim_catalogの3候補から重要な2件だけ選ぶ。JSONはclaims配列のみ。"
                    "各要素はcode, data_ids, confidence。ダミー語CODEは禁止。"
                    "data_idsは選んだcodeのsupported_byから1〜2件。"
                ),
            },
            ensure_ascii=False,
        )
        raw, parsed = ask_json(system, user, args.max_tokens)
        valid, rejected, seen_codes = [], [], set()
        for claim in (parsed or {}).get("claims", []):
            normalized, reason = validate_coded_claim(claim, ledger)
            if normalized and normalized["code"] not in seen_codes:
                normalized["origin"] = "model"
                valid.append(normalized)
                seen_codes.add(normalized["code"])
            else:
                rejected.append({"claim": claim, "reason": reason or "duplicate code"})
        for code in preferences.get(persona["id"], []):
            if len(valid) >= 3:
                break
            if code in catalog and code not in seen_codes:
                valid.append(
                    {
                        "code": code,
                        "data_ids": catalog[code]["supported_by"][:2],
                        "confidence": 60,
                        "origin": "validated_fallback",
                    }
                )
                seen_codes.add(code)
        persona_claims[persona["id"]] = valid
        run["independent"][persona["id"]] = {"raw": raw, "valid": valid, "rejected": rejected}
        print(f"\n[{persona['name']}] 採用={len(valid)} 失格={len(rejected)}", flush=True)
        for claim in valid:
            print(f"  {claim['code']} <- {','.join(claim['data_ids'])} ({claim['origin']})", flush=True)
        for item in rejected:
            print(f"  [失格] {item['reason']}", flush=True)

    events = schedule_claim_events(persona_claims, ledger, args.max_turns)
    run["events"] = events
    print("\n######## イベント駆動討論 / 異議・賛同を優先 ########", flush=True)
    persona_names = {persona["id"]: persona["name"] for persona in personas}
    for event in events:
        target = f" -> {event['target_claim_id']}" if event["target_claim_id"] else ""
        print(
            f"{event['claim_id']} [{event['action_label']}]{target} {persona_names[event['persona_id']]}: "
            f"{event['label']} [{','.join(event['data_ids'])}]",
            flush=True,
        )

    present_codes = {event["code"] for event in events}
    contested_pairs = sorted(
        {
            tuple(sorted((code, other)))
            for code in present_codes
            for other in catalog[code].get("contradicts", [])
            if other in present_codes
        }
    )
    previous_tally = {}
    for round_no in range(1, args.reconcile_rounds + 1):
        print(f"\n######## すり合わせ {round_no} / 対立ごとの一問一答 ########", flush=True)
        votes = {}
        for pair in contested_pairs:
            key = "|".join(pair)
            pair_votes = {}
            print(f"論点: {pair[0]} vs {pair[1]}", flush=True)
            for persona in personas:
                system = (
                    f"あなたは{persona['name']}。自由文は禁止。検証済み2コードの扱いを1語で選ぶ。"
                )
                user = json.dumps(
                    {
                        "left": {"code": pair[0], "label": catalog[pair[0]]["label"]},
                        "right": {"code": pair[1], "label": catalog[pair[1]]["label"]},
                        "own_initial_codes": [claim["code"] for claim in persona_claims[persona["id"]]],
                        "previous_tally": previous_tally.get(key, {}),
                        "rule": f"出力は {pair[0]} または {pair[1]} または BOTH または ABSTAIN の1語だけ。",
                    },
                    ensure_ascii=False,
                )
                raw, _ = ask_json(system, user, 48)
                matches = [code for code in pair if code in raw]
                if len(matches) == 1:
                    choice = matches[0]
                elif "BOTH" in raw or len(matches) == 2:
                    choice = "BOTH"
                else:
                    choice = "ABSTAIN"
                pair_votes[persona["id"]] = choice
                print(f"  {persona['name']}: {choice}", flush=True)
            votes[key] = pair_votes
        previous_tally = {
            key: {choice: list(pair_votes.values()).count(choice) for choice in set(pair_votes.values())}
            for key, pair_votes in votes.items()
        }
        run["reconciliation"].append({"round": round_no, "votes": votes, "tally": previous_tally})

    summary = synthesize_event_summary(events, catalog, run["reconciliation"], len(personas))
    run["summary"] = summary
    print("\n######## 検証済み統合 ########", flush=True)
    print("複数人格が合意:", ", ".join(summary["consensus"]) or "なし", flush=True)
    print("異議なしの検証済み主張:", ", ".join(summary["unopposed_supported"]) or "なし", flush=True)
    print("解決した対立:", summary["resolved_conflicts"] or "なし", flush=True)
    print("対立継続:", summary["unresolved_conflicts"] or "なし", flush=True)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"event_debate_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(outfile, run)
    print(f"保存: {outfile}", flush=True)
    return 0


def export_sft(args: argparse.Namespace) -> int:
    runs_path = Path(args.runs)
    grouped: dict[tuple[str, str], list[dict]] = {}
    source_runs: dict[tuple[str, str], set[str]] = {}
    skipped_unreviewed = 0
    for path in sorted(runs_path.rglob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if run.get("schema_version") != 1 or not isinstance(run.get("calls"), list):
            continue
        if run.get("training_review", {}).get("status") != "approved":
            skipped_unreviewed += 1
            continue
        domain = str(run.get("domain") or "unknown")
        if args.domain and domain != args.domain:
            continue
        approved_calls = set(run.get("training_review", {}).get("approved_calls", []))
        for call_index, call in enumerate(run["calls"]):
            if call_index not in approved_calls:
                continue
            if not all(call.get(key) for key in ("persona_id", "system", "user", "assistant")):
                continue
            key = (domain, str(call["persona_id"]))
            grouped.setdefault(key, []).append(
                {
                    "messages": [
                        {"role": "system", "content": call["system"]},
                        {"role": "user", "content": call["user"]},
                        {"role": "assistant", "content": call["assistant"]},
                    ]
                }
            )
            source_runs.setdefault(key, set()).add(str(run.get("run_id") or path.stem))

    if not grouped:
        print(f"承認済みの学習ログがありません。未承認/不採用スキップ: {skipped_unreviewed}", file=sys.stderr)
        return 1
    for (domain, persona_id), examples in grouped.items():
        target = Path(args.out) / domain / persona_id
        target.mkdir(parents=True, exist_ok=True)
        splits = split_examples(examples)
        for split, rows in splits.items():
            text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            (target / f"{split}.jsonl").write_text(text, encoding="utf-8")
        manifest = {
            "domain": domain,
            "persona_id": persona_id,
            "counts": {name: len(rows) for name, rows in splits.items()},
            "source_runs": sorted(source_runs[(domain, persona_id)]),
            "warning": "正解・根拠を検証したログだけを学習に採用してください。",
        }
        write_json(target / "manifest.json", manifest)
        print(f"{domain}/{persona_id}: {manifest['counts']}")
    return 0


def mark_run(args: argparse.Namespace) -> int:
    path = Path(args.run)
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("schema_version") != 1 or not isinstance(run.get("calls"), list):
        raise ValueError(f"CoD runではありません: {path}")
    approved_calls: list[int] = []
    if args.status == "approved":
        if not args.calls:
            raise ValueError("approvedには --calls 0,3,5 または --calls all が必要です")
        if args.calls == "all":
            approved_calls = list(range(len(run["calls"])))
        else:
            try:
                approved_calls = sorted({int(value) for value in args.calls.split(",")})
            except ValueError as exc:
                raise ValueError("--calls はカンマ区切りの番号です") from exc
            if any(index < 0 or index >= len(run["calls"]) for index in approved_calls):
                raise ValueError(f"call番号は0〜{len(run['calls']) - 1}です")
    run["training_review"] = {
        "status": args.status,
        "approved_calls": approved_calls,
        "reviewed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": args.note,
    }
    write_json(path, run)
    print(f"{path}: training_review={args.status}")
    return 0


def list_personas(args: argparse.Namespace) -> int:
    domains = load_domains()
    selected = {args.domain: domains[args.domain]} if args.domain else domains
    for domain, config in selected.items():
        print(f"{domain}: {config['description']}")
        for persona in config["personas"]:
            print(f"  {persona['id']}: {persona['name']} — {persona['worldview']}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)

    debate = subcommands.add_parser("debate", help="run blind opinions, cross-review, and moderation")
    debate.add_argument("topic")
    debate.add_argument("--domain", default="general", choices=sorted(load_domains()))
    debate.add_argument("--model", default=DEFAULT_MODEL)
    debate.add_argument("--api-url", default=DEFAULT_API_URL)
    debate.add_argument("--out", default="runs")
    debate.add_argument("--timeout", type=int, default=120)
    debate.add_argument("--num-predict", type=int, default=480)
    debate.add_argument("--seed", type=int, default=20260825)
    debate.set_defaults(handler=run_debate)

    export = subcommands.add_parser("export", help="export successful calls as per-persona MLX chat JSONL")
    export.add_argument("--runs", default="runs")
    export.add_argument("--out", default="data/sft")
    export.add_argument("--domain", choices=sorted(load_domains()))
    export.set_defaults(handler=export_sft)

    curriculum = subcommands.add_parser("curriculum", help="generate exact finite-set auditor SFT and holdout data")
    curriculum.add_argument("--out", default="data/auditor_curriculum")
    curriculum.add_argument("--count", type=int, default=240)
    curriculum.add_argument("--seed", type=int, default=20260825)
    curriculum.set_defaults(handler=make_auditor_curriculum)

    weight_eval = subcommands.add_parser("weight-eval", help="score base or LoRA Weight on the frozen auditor holdout")
    weight_eval.add_argument("--model", required=True)
    weight_eval.add_argument("--benchmark", default="data/auditor_curriculum/benchmark.jsonl")
    weight_eval.add_argument("--adapter")
    weight_eval.add_argument("--label", required=True)
    weight_eval.add_argument("--out", required=True)
    weight_eval.add_argument("--limit", type=int)
    weight_eval.add_argument("--max-tokens", type=int, default=420)
    weight_eval.set_defaults(handler=evaluate_weight)

    weight_audit = subcommands.add_parser("weight-audit", help="run the promoted tool-evidence-only auditor Weight")
    weight_audit.add_argument("--model", required=True)
    weight_audit.add_argument("--adapter", required=True)
    weight_audit.add_argument("--upper", type=int, required=True)
    weight_audit.add_argument("--condition", type=int, required=True)
    weight_audit.add_argument("--target", type=int, required=True)
    weight_audit.add_argument("--hypothesis", required=True)
    weight_audit.add_argument("--max-tokens", type=int, default=320)
    weight_audit.add_argument("--out")
    weight_audit.set_defaults(handler=run_weight_audit)

    event_debate = subcommands.add_parser("event-debate", help="run evidence-coded, event-driven MLX debate")
    event_debate.add_argument("--ledger", required=True)
    event_debate.add_argument("--domain", required=True, choices=sorted(load_domains()))
    event_debate.add_argument("--model-path", required=True)
    event_debate.add_argument("--out", default="runs")
    event_debate.add_argument("--max-turns", type=int, default=10)
    event_debate.add_argument("--reconcile-rounds", type=int, default=2)
    event_debate.add_argument("--max-tokens", type=int, default=240)
    event_debate.add_argument("--temperature", type=float, default=0.1)
    event_debate.set_defaults(handler=run_event_debate)

    mark = subcommands.add_parser("mark", help="explicitly approve or reject one run for training")
    mark.add_argument("run")
    mark.add_argument("status", choices=["approved", "rejected"])
    mark.add_argument("--calls", help="approved時に採用するcall番号。例: 2,5 または all")
    mark.add_argument("--note", required=True)
    mark.set_defaults(handler=mark_run)

    listing = subcommands.add_parser("list", help="list configured expert ensembles")
    listing.add_argument("--domain", choices=sorted(load_domains()))
    listing.set_defaults(handler=list_personas)
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        return args.handler(args)
    except KeyboardInterrupt:
        print("\n中断しました。進行済みデータは .partial.json に残っています。", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
