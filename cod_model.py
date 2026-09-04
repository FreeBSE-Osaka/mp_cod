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
import time
import unicodedata
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path


DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_API_URL = "http://127.0.0.1:11434/api/chat"
PROFILE_PATH = Path(__file__).with_name("personas.json")
STANCES = ["主案", "対案", "条件付き", "保留"]
EVENT_PROMPT_PROFILES = ("baseline", "orthogonal", "orthogonal_bare", "orthogonal_fewshot")
MECHANICAL_UTTERANCE_PHRASES = (
    "この点を今後の判断の軸",
    "現時点では、",
    "という可能性を重く見ています",
    "という展開も考えておくべき",
    "という選択も検討中",
    "と判断します。根拠は",
    "という点を見落としていました。見方を改めます",
    "ことが判断です",
)
MODEL_UTTERANCE_ORIGINS = {
    "model",
    "model_reaction",
    "model_repair",
    "model_sanitized",
    "model_dialogue_v2",
    "model_dialogue_v2_repair",
    "model_renderer_v3",
    "model_renderer_v3_sanitized",
    "model_body_v1",
    "model_body_v2",
}
MOVE_UTTERANCE_TEMPLATES = {
    "object": (
        "ただ、その案には懸念があります。代わりに、『{label}』を先に試すべきです。",
        "その結論には異議があります。まず『{label}』で条件を確かめる案を提案します。",
        "そのまま広げるのは危険です。『{label}』を採用条件として先に検証したいです。",
        "問題を避けるなら、一度『{label}』へ修正して進めるべきです。",
    ),
    "agree": (
        "その案には賛成です。私としては、『{label}』という点も大事だと思います。",
        "同じ結論です。『{label}』を理由に加えたいです。",
        "そこは同意します。特に『{label}』は外せません。",
    ),
    "maintain": (
        "結論は変わりません。私は『{label}』を支持します。",
        "今のところ見方は同じです。『{label}』が決め手です。",
        "結論を維持します。再確認しても『{label}』を選びます。",
    ),
    "revise": (
        "前の見方を修正します。『{label}』の方がデータに合っています。",
        "考え直しました。今回は『{label}』を支持します。",
        "先ほどとは結論を変えます。『{label}』を採ります。",
    ),
}
MOVE_UTTERANCE_PREFIXES = {
    "object": (
        "ただ、その案には懸念があります。代わりに、",
        "その結論には異議があります。代案として、",
        "そのまま進めるのは危険です。まず、",
        "問題を避けるなら、",
    ),
    "agree": (
        "その案には賛成です。加えて、",
        "同意します。私の観点では、",
        "私も賛成です。特に、",
    ),
    "maintain": (
        "結論は変わりません。",
        "今のところ見方は同じです。",
        "この判断を維持します。",
    ),
    "revise": (
        "前の見方を修正します。",
        "考え直しました。",
        "先ほどとは結論を変えます。",
    ),
}
BODY_RENDERER_SYSTEM = (
    "各itemのspeakerとして、検証済みclaimを自然な日本語一文で述べる本文renderer。"
    "claimの内容、時制、数字を変更・追加せず、moveや賛否は表現しない。"
    "入力itemsと同じidを一度ずつ返す。"
    "出力はbodiesだけをキーに持つJSONで、各要素のキーはidとbodyだけ。"
    "必ず {\"bodies\":[{\"id\":\"入力id\",\"body\":\"本文\"}]} の形で返し、"
    "idをJSONキーにしてはならない。"
    "提案や計画を実現済み・検証済み等の完了事実へ変えず、claimの時制と確実性を保つ。"
)
BODY_MOVE_MARKERS = (
    "その案には賛成", "その案に賛成", "私も賛同", "私も同意", "その結論には異議",
    "その案には異議", "その案には懸念", "結論を維持", "見方を修正", "考え直しました",
    "結論を変え", "を採ります", "を選びます",
)
BODY_MODALITY_SHIFT_MARKERS = (
    "検証済み", "実現", "完了", "達成", "導入済み", "展開済み", "実施済み",
    "展開した", "実施した", "導入した", "開始した", "作成した", "提案した",
    "検討している", "予定している", "推定され", "見込まれ",
)
BODY_FRAGMENT_ENDINGS = (
    "ことを提案。", "ことを検討。", "ことを監査。", "ことを評価。",
)


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
            "rebuttal_type": {
                "type": "string",
                "enum": ["前提の否定", "反例の提示", "トレードオフの指摘"],
            },
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
            "rebuttal_type",
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
            required = ("name", "worldview", "objective", "utility", "loss", "tests", "avoid")
            if any(not persona.get(key) for key in required):
                raise ValueError(f"{domain}/{persona['id']}: missing persona field")
    return domains


def persona_system(persona: dict, phase: str) -> str:
    return f"""あなたは同一基盤モデル内の独立した専門家人格「{persona['name']}」です。
世界観: {persona['worldview']}
目的: {persona['objective']}
最大化する効用: {persona['utility']}
最小化する損失: {persona['loss']}
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
    schema: dict | str,
    api_url: str,
    timeout: int,
    num_predict: int,
    temperature: float,
    seed: int,
    include_raw: bool = False,
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
            if include_raw:
                meta["_raw_content"] = content
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
反論型は「前提の否定」「反例の提示」「トレードオフの指摘」の一つを選びます。
短い例: 前提の否定なら「対象案は常時接続を前提にしているが、要件にはオフライン利用がある」のように、
相手の文章の言い換えではなく、壊れる前提・反例・代償を一つ特定してください。
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


def batch_renderer_examples(rows: list[dict], persona: dict, batch_size: int) -> list[dict]:
    batched = []
    for start in range(0, len(rows), batch_size):
        items, utterances = [], []
        for index, row in enumerate(rows[start : start + batch_size], 1):
            payload = json.loads(row["messages"][1]["content"])
            if payload.get("phase") == "event":
                payload["move"] = renderer_event_move(payload.get("move"))
            payload["speech_act"] = renderer_move_instruction(payload.get("move"))
            item_id = f"I{index:02d}"
            items.append({"id": item_id, **payload})
            answer = json.loads(row["messages"][2]["content"])
            utterances.append({"id": item_id, "utterance": answer["utterance"]})
        batched.append(
            {
                "messages": [
                    {"role": "system", "content": renderer_system(persona)},
                    {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
                    {
                        "role": "assistant",
                        "content": json.dumps({"utterances": utterances}, ensure_ascii=False),
                    },
                ]
            }
        )
    return batched


def batch_shared_renderer_examples(
    rows: list[tuple[dict, dict]], batch_size: int
) -> list[dict]:
    batched = []
    for start in range(0, len(rows), batch_size):
        items, utterances = [], []
        for index, (persona, row) in enumerate(rows[start : start + batch_size], 1):
            payload = json.loads(row["messages"][1]["content"])
            if payload.get("phase") == "event":
                payload["move"] = renderer_event_move(payload.get("move"))
            payload["speech_act"] = renderer_move_instruction(payload.get("move"))
            item_id = f"I{index:02d}"
            items.append(
                {
                    "id": item_id,
                    **payload,
                    "speaker": persona["name"],
                }
            )
            answer = json.loads(row["messages"][2]["content"])
            utterances.append({"id": item_id, "utterance": answer["utterance"]})
        batched.append(
            {
                "messages": [
                    {"role": "system", "content": renderer_system(None)},
                    {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
                    {
                        "role": "assistant",
                        "content": json.dumps({"utterances": utterances}, ensure_ascii=False),
                    },
                ]
            }
        )
    return batched


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


def load_adapter_map(path: str | None, persona_ids: set[str]) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    adapters = payload.get("adapters") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(adapters, dict):
        raise ValueError("adapter map requires schema_version=1 and an adapters object")
    unknown = set(adapters) - persona_ids
    if unknown:
        raise ValueError(f"adapter map has unknown personas: {sorted(unknown)}")
    result = {}
    for persona_id, adapter_path in adapters.items():
        if not isinstance(adapter_path, str) or not Path(adapter_path).is_dir():
            raise ValueError(f"{persona_id}: adapter directory does not exist: {adapter_path}")
        for required in ("adapter_config.json", "adapters.safetensors"):
            if not (Path(adapter_path) / required).is_file():
                raise ValueError(f"{persona_id}: missing {required}")
        result[persona_id] = adapter_path
    return result


def label_statement(code: str, data_ids: list[str], ledger: dict) -> str:
    catalog = {item["code"]: item for item in ledger["claim_catalog"]}
    return f"{catalog[code]['label']}。根拠は[{','.join(data_ids)}]。"


def validate_public_statement(statement: object, data_ids: list[str]) -> tuple[str | None, str | None]:
    if not isinstance(statement, str):
        return None, "statement must be a string"
    normalized = re.sub(r"\s+", " ", statement).strip()
    if not 8 <= len(normalized) <= 240:
        return None, "statement must be 8 to 240 characters"
    if any(mark in normalized for mark in ("空文字", "JSONキー", "changed=true", "allowed_data_ids")):
        return None, "statement exposes internal protocol"
    references = set(re.findall(r"D\d{2,}", normalized))
    if not references:
        return None, "statement must cite at least one D id"
    if not references.issubset(set(data_ids)):
        return None, f"statement cites unselected D ids: {sorted(references - set(data_ids))}"
    return normalized, None


def sanitize_model_statement(statement: object, data_ids: list[str]) -> str | None:
    if not isinstance(statement, str) or not data_ids:
        return None
    normalized = re.sub(r"\s+", " ", statement).strip()
    corrected = re.sub(r"\[\s*D\d{2,}\s*\]", f"[{data_ids[0]}]", normalized)
    corrected, _ = validate_public_statement(corrected, data_ids)
    if corrected is not None:
        return corrected
    if "根拠" in normalized:
        normalized = normalized.split("根拠", 1)[0]
    normalized = re.sub(r"\[?D\d{2,}\]?", "", normalized)
    normalized = normalized.strip(" 、,。.[]")
    if not 8 <= len(normalized) <= 200 or not re.search(r"[ぁ-んァ-ヶ一-龠]", normalized):
        return None
    return f"{normalized}。根拠は[{','.join(data_ids)}]。"


def validate_dialogue_utterance(utterance: object) -> tuple[str | None, str | None]:
    if not isinstance(utterance, str):
        return None, "utterance must be a string"
    normalized = re.sub(r"\s+", " ", utterance).strip()
    if not 12 <= len(normalized) <= 320:
        return None, "utterance must be 12 to 320 characters"
    if re.search(r"D\d{2,}", normalized):
        return None, "utterance must keep D ids in metadata, not dialogue"
    if "_" in normalized or any(mark in normalized for mark in ("{", "}", "claim_catalog", "data_ids")):
        return None, "utterance exposes internal protocol"
    if re.search(r"[。！？][のをにがは](?=[ぁ-んァ-ヶ一-龠])", normalized):
        return None, "utterance contains an orphan particle after a sentence boundary"
    if not re.search(r"[ぁ-んァ-ヶ一-龠]", normalized):
        return None, "utterance must contain natural Japanese"
    if normalized[-1] not in "。！？!?":
        return None, "utterance must end as a complete sentence"
    if re.search(r"(?:を|へ|に|として|について|条件を)[。！？!?]$", normalized):
        return None, "utterance must end as a complete sentence"
    return normalized, None


def renderer_system(persona: dict | None) -> str:
    if persona is None:
        identity = "各itemのspeakerとして発言する。"
    else:
        utility = str(persona["utility"]).rstrip("。")
        loss = str(persona["loss"]).rstrip("。")
        identity = (
            f"あなたは{persona['name']}。最大化する効用: {utility}。"
            f"最小化する損失: {loss}。"
        )
    return (
        f"{identity}検証済みの構造化判断を会話文へ描画する専用rendererである。"
        "主張、選択、根拠、moveを変更・再評価・追加してはならない。"
        "objectは異議に加えて代案・修正版・採用条件のいずれかを述べる。"
        "agreeは賛同に自分の観点を加え、maintainは維持理由、reviseは判断変更を率直に述べる。"
        "code、D番号、内部キーを読み上げず、12〜320字の自然な日本語で完結させる。"
        "入力itemsと同じidを順不同で一度ずつ返す。"
        "出力はutterancesだけをキーに持つJSONで、各要素のキーはidとutteranceだけ。"
    )


def renderer_event_move(action: object) -> str:
    return {"new": "propose", "object": "object", "agree_extend": "agree"}.get(
        str(action), "propose"
    )


def renderer_move_instruction(move: object) -> str:
    return {
        "propose": "own_claimを自分の提案として述べる。",
        "object": "target_claimへ異議を示し、own_claimを代案・修正版・採用条件として述べる。",
        "agree": "相手へ賛同し、own_claimを自分の観点として加える。",
        "maintain": "判断を維持すると明言し、selected_claimを理由とともに述べる。",
        "revise": "前の判断を変えると明言し、selected_claimを新しい選択として述べる。",
    }.get(str(move), "入力済みの主張をそのまま自然に述べる。")


RENDERER_RESTRICTIONS = (
    "ない", "ません", "禁止", "不可", "避け", "控え", "限定", "留め", "のみ", "だけ", "専用",
    "危険", "問題", "反対", "異議",
)
RENDERER_POSITIVE_ACTIONS = {
    "置換": r"置換(?:する|します|しよう|を行う)",
    "採用": r"採用(?:する|します|しよう)",
    "導入": r"導入(?:する|します|しよう)",
    "昇格": r"昇格(?:する|します|させる)",
    "公開": r"公開(?:する|します|しよう)",
    "実施": r"実施(?:する|します|しよう)",
    "使用": r"使用(?:する|します|しよう)",
    "使": r"使(?:う|います|おう)",
}


def _restriction_near(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 6) : end + 12]
    return any(marker in window for marker in RENDERER_RESTRICTIONS + ("却下",))


def _restricted_action_stems(label: str) -> set[str]:
    return {
        stem
        for stem in RENDERER_POSITIVE_ACTIONS
        if any(_restriction_near(label, match.start(), match.end()) for match in re.finditer(re.escape(stem), label))
    }


def dialogue_selects_competing_claim(utterance: str, competitors: list[str]) -> bool:
    for label in competitors:
        if not label:
            continue
        if re.search(rf"{re.escape(label)}[』\s]*(?:を|へ)(?:選|採|支持)", utterance):
            return True
        shared_directive = any(marker in label and marker in utterance for marker in ("直ちに", "すぐ", "自動", "全面", "本番"))
        if not shared_directive:
            continue
        for stem, pattern in RENDERER_POSITIVE_ACTIONS.items():
            if stem not in label:
                continue
            if any(not _restriction_near(utterance, match.start(), match.end()) for match in re.finditer(pattern, utterance)):
                return True
    return False


def dialogue_reverses_restriction(utterance: str, label: str) -> bool:
    for stem in _restricted_action_stems(label):
        pattern = RENDERER_POSITIVE_ACTIONS[stem]
        for match in re.finditer(pattern, utterance):
            if not _restriction_near(utterance, match.start(), match.end()):
                return True
    return False


def dialogue_preserves_restriction(utterance: str, label: str) -> bool:
    restricted = _restricted_action_stems(label)
    return not restricted or any(
        _restriction_near(utterance, match.start(), match.end())
        for stem in restricted
        for match in re.finditer(re.escape(stem), utterance)
    )


def parse_renderer_utterances(
    parsed: object, expected_ids: list[str]
) -> tuple[dict[str, object], str | None]:
    if not isinstance(parsed, dict) or set(parsed) != {"utterances"}:
        return {}, "renderer output must contain only utterances"
    rows = parsed.get("utterances")
    if not isinstance(rows, list):
        return {}, "renderer utterances must be an array"
    expected = set(expected_ids)
    values: dict[str, object] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "utterance"}:
            return {}, "renderer rows must contain only id and utterance"
        item_id = row.get("id")
        if not isinstance(item_id, str) or item_id not in expected or item_id in values:
            return {}, "renderer returned an unknown or duplicate id"
        values[item_id] = row.get("utterance")
    if set(values) != expected:
        return {}, "renderer did not return every requested id"
    return values, None


def parse_renderer_bodies(
    parsed: object, expected_ids: list[str]
) -> tuple[dict[str, object], str | None, bool]:
    if isinstance(parsed, dict) and set(parsed) == {"bodies"}:
        rows = parsed.get("bodies")
        if not isinstance(rows, list):
            return {}, "renderer bodies must be an array", False
        expected = set(expected_ids)
        values: dict[str, object] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "body"}:
                return {}, "renderer body rows must contain only id and body", False
            item_id = row.get("id")
            if not isinstance(item_id, str) or item_id not in expected or item_id in values:
                return {}, "renderer returned an unknown or duplicate body id", False
            values[item_id] = row.get("body")
        if set(values) != expected:
            return {}, "renderer did not return every requested body id", False
        return values, None, False
    if (
        len(expected_ids) == 1
        and isinstance(parsed, dict)
        and set(parsed) == {expected_ids[0]}
        and isinstance(parsed[expected_ids[0]], str)
    ):
        return {expected_ids[0]: parsed[expected_ids[0]]}, "normalized single-id body mapping", True
    return {}, "renderer output must contain only bodies", False


def normalize_renderer_body(body: object) -> tuple[str | None, str | None]:
    if not isinstance(body, str):
        return None, "body must be a string"
    normalized = re.sub(r"\s+", " ", body).strip()
    if normalized and normalized[-1] not in "。！？!?":
        normalized += "。"
    return validate_dialogue_utterance(normalized)


def body_is_neutral(body: str) -> bool:
    return not any(marker in body for marker in BODY_MOVE_MARKERS)


def body_is_polite_sentence(body: str) -> bool:
    return body.rstrip("。！？!?").endswith(("です", "ます", "ません", "でした", "ました"))


def body_matches_claim(body: str, label: str) -> bool:
    return (
        dialogue_matches_claim(body, label)
        and not dialogue_reverses_restriction(body, label)
        and dialogue_preserves_restriction(body, label)
        and not any(marker in body and marker not in label for marker in BODY_MODALITY_SHIFT_MARKERS)
        and not body.endswith(BODY_FRAGMENT_ENDINGS)
    )


def event_execution_settings(args: argparse.Namespace, persona_count: int) -> dict[str, int]:
    if not getattr(args, "fast", False):
        return {
            "claims_per_persona": 3,
            "max_turns": args.max_turns,
            "reconcile_rounds": args.reconcile_rounds,
            "decision_max_tokens": args.max_tokens,
        }
    # ponytail: fast mode omits model reconciliation; use full mode when forced consensus is required.
    return {
        "claims_per_persona": 2,
        "max_turns": min(args.max_turns, persona_count * 2),
        "reconcile_rounds": 0,
        "decision_max_tokens": min(args.max_tokens, 320),
    }


def restore_claim_label(utterance: object, label: str) -> object:
    if not isinstance(utterance, str) or label in utterance or len(label) < 12:
        return utterance
    collapsed = label.replace("する", "る")
    if collapsed != label and collapsed in utterance:
        return utterance.replace(collapsed, label, 1)
    width = len(label)
    candidates = (
        (SequenceMatcher(None, label, utterance[start : start + width], autojunk=False).ratio(), start)
        for start in range(max(0, len(utterance) - width + 1))
    )
    best_score, start = max(candidates, default=(0.0, 0))
    if best_score < 0.92:
        return utterance
    fragment = list(utterance[start : start + width])
    typos = [index for index, (expected, actual) in enumerate(zip(label, fragment)) if expected != actual and index < width - 3]
    if not 1 <= len(typos) <= 2:
        return utterance
    for index in typos:
        fragment[index] = label[index]
    return f"{utterance[:start]}{''.join(fragment)}{utterance[start + width:]}"


def dialogue_move_example(label: str, move: str, variant: int = 0) -> str:
    templates = MOVE_UTTERANCE_TEMPLATES[move]
    return templates[variant % len(templates)].replace("{label}", label)


def validate_dialogue_move(utterance: object, move: str) -> tuple[str | None, str | None]:
    normalized, reason = validate_dialogue_utterance(utterance)
    if normalized is None:
        return None, reason
    if move == "propose" and any(
        marker in normalized
        for marker in (
            "その案", "賛成", "賛同", "同意", "同感", "前の見方", "判断を変え", "結論を維持",
        )
    ):
        return None, "propose must not pretend to answer or revise another claim"
    markers = {
        "object": (
            "いえ", "ただ", "しかし", "その見方", "見落と", "抜け", "懸念",
            "不十分", "問題", "欠陥", "反対", "異議", "危険", "難しい", "許容範囲を超え",
        ),
        "agree": ("賛成", "賛同", "同意", "同感", "支持", "私も", "同じ見方", "同じ結論"),
        "maintain": (
            "維持", "引き続き", "変わりません", "変えません", "同じ判断", "見方です", "見方は同じ",
        ),
        "revise": ("確かに", "見落と", "改め", "修正", "変更", "切り替え", "考え直", "結論を変え"),
    }
    required = markers.get(move)
    if required and not any(marker in normalized for marker in required):
        return None, f"utterance does not express dialogue move: {move}"
    if move == "object" and not any(
        marker in normalized
        for marker in ("代わり", "代案", "提案", "条件", "なら", "まず", "先に", "修正", "べき", "した上で", "案を")
    ):
        return None, "objection must include an alternative, revision, or adoption condition"
    return normalized, None


def is_mechanical_utterance(utterance: str) -> bool:
    return any(phrase in utterance for phrase in MECHANICAL_UTTERANCE_PHRASES)


def reaction_is_aligned(utterance: str, own_label: str, target_label: str, action: str) -> bool:
    if not dialogue_is_aligned(utterance, own_label, [target_label]):
        return False
    if action != "object":
        return True
    return not dialogue_selects_competing_claim(utterance, [target_label])


def dialogue_matches_claim(utterance: str, label: str) -> bool:
    if label in utterance or similarity(utterance, label) >= 0.3:
        return True
    utterance_ngrams = normalized_ngrams(utterance)
    label_ngrams = normalized_ngrams(label)
    directional_coverage = (
        len(utterance_ngrams & label_ngrams) / len(label_ngrams) if label_ngrams else 0.0
    )
    if directional_coverage >= 0.45:
        return True
    salient = set(re.findall(r"[a-z][a-z0-9._+-]*|\d+(?:\.\d+)?%?", label.casefold()))
    numeric = {token for token in salient if any(character.isdigit() for character in token)}
    matched = {token for token in salient if token in utterance.casefold()}
    return bool(numeric) and len(matched) >= 2 and len(matched) / len(salient) >= 0.6


def dialogue_is_aligned(utterance: str, label: str, competitors: list[str]) -> bool:
    if dialogue_reverses_restriction(utterance, label) or not dialogue_preserves_restriction(utterance, label):
        return False
    if dialogue_matches_claim(utterance, label):
        return True
    selected = similarity(utterance, label)
    competing = max((similarity(utterance, value) for value in competitors), default=0.0)
    return selected >= 0.04 and selected >= competing + 0.02


def dialogue_numbers_are_grounded(utterance: str, payload: dict) -> bool:
    numbers = lambda value: set(
        re.findall(r"\d+(?:\.\d+)?", unicodedata.normalize("NFKC", value))
    )
    source = json.dumps(payload, ensure_ascii=False)
    return numbers(utterance).issubset(numbers(source))


def independent_utterance_is_aligned(utterance: str, code: str, claims: list[dict]) -> bool:
    scores = {claim["code"]: similarity(utterance, claim["label"]) for claim in claims}
    return code in scores and scores[code] >= max(scores.values(), default=0.0)


def dialogue_fallback(statement: str) -> str:
    visible = statement.split("根拠", 1)[0]
    visible = re.sub(r"\[?D\d{2,}\]?", "", visible).strip(" 、,。.[]")
    return f"{visible}。"


def compose_dialogue_body(body: str, label: str, move: str, variant: int = 0) -> str | None:
    if body.startswith(label) and body[len(label) :].startswith("を"):
        body = f"『{label}』{body[len(label):]}"
    prefixes = MOVE_UTTERANCE_PREFIXES.get(move)
    candidate = f"{prefixes[variant % len(prefixes)]}{body}" if prefixes else body
    normalized, _ = validate_dialogue_move(candidate, move)
    return normalized if normalized is not None and dialogue_is_aligned(normalized, label, []) else None


def compose_dialogue_fallback(
    statement: str, label: str, move: str, variant: int = 0
) -> tuple[str, str]:
    body = dialogue_fallback(statement)
    composed = compose_dialogue_body(body, label, move, variant)
    if composed is not None:
        return composed, "composed_statement_fallback"
    if move in MOVE_UTTERANCE_TEMPLATES:
        return dialogue_move_example(label, move, variant), "template_fallback"
    return body, "statement_fallback"


def sanitize_dialogue_move(utterance: object, move: str) -> str | None:
    if not isinstance(utterance, str):
        return None
    visible = utterance.split("根拠", 1)[0]
    visible = re.sub(r"\[?D\d{2,}\]?", "", visible).strip(" 、,。.[]")
    visible = re.sub(r"([。！？])の(結果|データ)", r"\1その\2", visible)
    visible = re.sub(r"^の(結果|データ)", r"その\1", visible)
    if not visible:
        return None
    visible = f"{visible}。"
    normalized, _ = validate_dialogue_move(visible, move)
    if normalized is not None:
        return normalized
    if move == "object" and visible.startswith("その案には、"):
        visible = visible.removeprefix("その案には、")
    if move == "object" and any(
        marker in visible for marker in ("賛成", "賛同", "同意", "同感", "支持")
    ) and not any(marker in visible for marker in ("ただ", "しかし", "懸念", "異議", "不十分")):
        return None
    if move == "agree" and any(
        marker in visible for marker in ("賛成", "賛同", "同意", "同感", "支持")
    ):
        return None
    prefixes = {
        "object": "その案には懸念があります。代わりに、",
        "agree": "その案に賛成です。",
        "maintain": "結論は変わりません。",
        "revise": "前の見方を修正します。",
    }
    candidate = f"{prefixes.get(move, '')}{visible}"
    normalized, _ = validate_dialogue_move(candidate, move)
    return normalized


def validate_coded_claim(claim: dict, ledger: dict) -> tuple[dict | None, str | None]:
    if not isinstance(claim, dict):
        return None, "claim is not an object"
    code = claim.get("code")
    catalog = {item["code"]: item for item in ledger["claim_catalog"]}
    if code not in catalog:
        return None, f"unknown claim code: {code}"
    data_ids = claim.get("data_ids")
    if (
        not isinstance(data_ids, list)
        or not data_ids
        or not all(isinstance(value, str) for value in data_ids)
    ):
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


def prior_pair_choice(claims: list[dict], pair: tuple[str, str], prior_vote: dict) -> str | None:
    if prior_vote.get("choice"):
        return str(prior_vote["choice"])
    initial = [claim["code"] for claim in claims if claim.get("code") in pair]
    return initial[0] if len(initial) == 1 else None


def reconciliation_has_supermajority(
    tally: dict[str, dict[str, int]], pairs: list[tuple[str, str]], persona_count: int
) -> bool:
    threshold = max(1, (persona_count * 3 + 3) // 4)
    return bool(pairs) and all(
        any(tally.get("|".join(pair), {}).get(code, 0) >= threshold for code in pair)
        for pair in pairs
    )


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
            "origin": claim.get("origin", "unknown"),
            "statement": claim.get("statement") or label_statement(claim["code"], claim["data_ids"], ledger),
            "statement_origin": claim.get("statement_origin", "label_fallback"),
            "utterance": claim.get("utterance") or dialogue_fallback(
                claim.get("statement") or label_statement(claim["code"], claim["data_ids"], ledger)
            ),
            "utterance_origin": claim.get("utterance_origin", "statement_fallback"),
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
            normalized_choices = [
                choice.get("choice") if isinstance(choice, dict) else choice for choice in choices.values()
            ]
            counts = {code: sum(choice == code for choice in normalized_choices) for code in pair}
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


def event_run_metrics(run: dict) -> dict:
    events = run.get("events") or []
    independent = run.get("independent") or {}
    model_claims = sum(event.get("origin") == "model" for event in events)
    fallbacks = sum(event.get("origin") == "validated_fallback" for event in events)
    model_statement_origins = {"model", "model_repair", "model_sanitized"}
    model_utterance_origins = MODEL_UTTERANCE_ORIGINS
    body_utterance_origins = {
        "model_body_v1", "model_body_v1_schema_repair",
        "model_body_v2", "model_body_v2_schema_repair",
    }
    model_statements = sum(event.get("statement_origin") in model_statement_origins for event in events)
    model_event_utterances = sum(event.get("utterance_origin") in model_utterance_origins for event in events)
    body_event_utterances = sum(event.get("utterance_origin") in body_utterance_origins for event in events)
    dialogue_v2_utterances = sum(
        event.get("utterance_origin") in {"model_dialogue_v2", "model_dialogue_v2_repair"}
        for event in events
    )
    dialogue_v3_utterances = sum(
        event.get("utterance_origin") in {"model_renderer_v3", "model_renderer_v3_sanitized"}
        for event in events
    )
    reaction_events = [event for event in events if event.get("action") in {"object", "agree_extend"}]
    reaction_failures = sum(
        event.get("utterance_origin") not in model_utterance_origins
        for event in reaction_events
    )
    rejected = sum(len(value.get("rejected") or []) for value in independent.values())
    model_attempts = model_claims + rejected
    raw_records = [value.get("raw") for value in independent.values()]
    raw_records.extend(
        value.get("dialogue_render_raw")
        for value in independent.values()
        if value.get("dialogue_render_raw") is not None
    )
    raw_records.extend(
        value.get("dialogue_render_repair_raw")
        for value in independent.values()
        if value.get("dialogue_render_repair_raw") is not None
    )
    raw_records.extend(event.get("reaction_raw") for event in events if event.get("reaction_raw") is not None)
    raw_records.extend(
        event.get("reaction_repair_raw") for event in events if event.get("reaction_repair_raw") is not None
    )
    raw_records.extend(
        batch.get("raw") for batch in run.get("renderer_batches") or [] if batch.get("raw") is not None
    )
    vote_records = []
    for round_data in run.get("reconciliation") or []:
        for pair_votes in (round_data.get("votes") or {}).values():
            for vote in pair_votes.values():
                if isinstance(vote, dict):
                    vote_records.append(vote)
                    raw_records.append(vote.get("raw"))
                    if vote.get("repair_raw") is not None:
                        raw_records.append(vote.get("repair_raw"))
    raw_record_rate = sum(isinstance(value, str) and bool(value.strip()) for value in raw_records) / len(raw_records) if raw_records else 0.0
    statement_pairs = [
        similarity(str(left.get("statement", "")), str(right.get("statement", "")))
        for left, right in itertools.combinations(events, 2)
        if left.get("code") != right.get("code")
    ]
    near_duplicates = sum(value >= 0.8 for value in statement_pairs)
    total = len(events)
    model_claim_rate = model_claims / total if total else 0.0
    fallback_rate = fallbacks / total if total else 0.0
    vote_model_statements = sum(
        vote.get("statement_origin") in model_statement_origins for vote in vote_records
    )
    vote_model_utterances = sum(
        vote.get("utterance_origin") in model_utterance_origins for vote in vote_records
    )
    body_vote_utterances = sum(
        vote.get("utterance_origin") in body_utterance_origins for vote in vote_records
    )
    public_statement_total = total + len(vote_records)
    model_statement_rate = (
        (model_statements + vote_model_statements) / public_statement_total if public_statement_total else 0.0
    )
    model_utterance_rate = (
        (model_event_utterances + vote_model_utterances) / public_statement_total
        if public_statement_total
        else 0.0
    )
    body_utterance_rate = (
        (body_event_utterances + body_vote_utterances) / public_statement_total
        if public_statement_total
        else 0.0
    )
    dialogue_records = [(event.get("utterance", ""), event.get("code")) for event in events] + [
        (vote.get("utterance", ""), vote.get("choice")) for vote in vote_records
    ]
    dialogue_pairs = [
        similarity(left_text, right_text)
        for (left_text, left_code), (right_text, right_code) in itertools.combinations(dialogue_records, 2)
        if left_code != right_code
    ]
    dialogue_near_duplicates = sum(value >= 0.8 for value in dialogue_pairs)
    dialogue_near_duplicate_rate = (
        dialogue_near_duplicates / len(dialogue_pairs) if dialogue_pairs else 0.0
    )
    mechanical_utterances = sum(is_mechanical_utterance(text) for text, _ in dialogue_records)
    mechanical_utterance_rate = mechanical_utterances / len(dialogue_records) if dialogue_records else 0.0
    vote_parse_fallbacks = sum(vote.get("choice_origin") == "parse_fallback" for vote in vote_records)
    changed_votes = [vote for vote in vote_records if vote.get("changed_from_previous")]
    model_change_reasons = sum(
        vote.get("change_reason_origin") in model_statement_origins for vote in changed_votes
    )
    change_reason_rate = model_change_reasons / len(changed_votes) if changed_votes else 1.0
    distinct_code_rate = len({event.get("code") for event in events}) / total if total else 0.0
    action_diversity = len({event.get("action") for event in events}) / len(ACTION_PRIORITY) if events else 0.0
    near_duplicate_rate = near_duplicates / len(statement_pairs) if statement_pairs else 0.0
    evidence_validity_rate = model_claims / model_attempts if model_attempts else 0.0
    score = 100 * (
        0.15 * model_claim_rate
        + 0.1 * model_statement_rate
        + 0.15 * model_utterance_rate
        + 0.15 * raw_record_rate
        + 0.1 * distinct_code_rate
        + 0.05 * action_diversity
        + 0.1 * (1.0 - near_duplicate_rate)
        + 0.1 * (1.0 - dialogue_near_duplicate_rate)
        + 0.1 * (1.0 - mechanical_utterance_rate)
    )
    return {
        "event_count": total,
        "model_claims": model_claims,
        "validated_fallbacks": fallbacks,
        "rejected_claims": rejected,
        "model_claim_rate": round(model_claim_rate, 4),
        "fallback_rate": round(fallback_rate, 4),
        "evidence_validity_rate": round(evidence_validity_rate, 4),
        "model_statement_rate": round(model_statement_rate, 4),
        "event_model_statements": model_statements,
        "reconciliation_model_statements": vote_model_statements,
        "model_utterance_rate": round(model_utterance_rate, 4),
        "event_model_utterances": model_event_utterances,
        "body_model_utterance_rate": round(body_utterance_rate, 4),
        "body_model_utterances": body_event_utterances + body_vote_utterances,
        "body_schema_repairs": sum(
            text.get("utterance_origin") in {"model_body_v1_schema_repair", "model_body_v2_schema_repair"}
            for text in events + vote_records
        ),
        "dialogue_v2_utterances": dialogue_v2_utterances,
        "dialogue_v3_utterances": dialogue_v3_utterances,
        "reaction_events": len(reaction_events),
        "reaction_failures": reaction_failures,
        "reconciliation_model_utterances": vote_model_utterances,
        "dialogue_near_duplicate_pairs": dialogue_near_duplicates,
        "dialogue_near_duplicate_rate": round(dialogue_near_duplicate_rate, 4),
        "mechanical_utterances": mechanical_utterances,
        "mechanical_utterance_rate": round(mechanical_utterance_rate, 4),
        "reconciliation_parse_fallbacks": vote_parse_fallbacks,
        "changed_votes": len(changed_votes),
        "model_change_reason_rate": round(change_reason_rate, 4),
        "raw_record_rate": round(raw_record_rate, 4),
        "distinct_code_rate": round(distinct_code_rate, 4),
        "action_diversity": round(action_diversity, 4),
        "near_duplicate_pairs": near_duplicates,
        "near_duplicate_rate": round(near_duplicate_rate, 4),
        "shadow_score": round(score, 2),
        "hard_gate_pass": (
            rejected == 0
            and evidence_validity_rate == 1.0
            and raw_record_rate == 1.0
            and vote_parse_fallbacks == 0
            and change_reason_rate == 1.0
            and model_statement_rate == 1.0
            and model_utterance_rate == 1.0
            and reaction_failures == 0
        ),
    }


def decide_rsi_shadow(
    parent_dev: dict,
    candidate_dev: dict,
    parent_holdout: dict,
    candidate_holdout: dict,
    *,
    round_no: int,
    max_rounds: int,
    holdout_distinct: bool,
) -> dict:
    dev_gain = candidate_dev["shadow_score"] - parent_dev["shadow_score"]
    holdout_gain = candidate_holdout["shadow_score"] - parent_holdout["shadow_score"]
    non_regression = all(
        (
            candidate_dev["fallback_rate"] <= parent_dev["fallback_rate"],
            candidate_holdout["fallback_rate"] <= parent_holdout["fallback_rate"],
            candidate_dev["model_statement_rate"] >= parent_dev["model_statement_rate"],
            candidate_holdout["model_statement_rate"] >= parent_holdout["model_statement_rate"],
            candidate_dev.get("model_utterance_rate", 0.0) >= parent_dev.get("model_utterance_rate", 0.0),
            candidate_holdout.get("model_utterance_rate", 0.0)
            >= parent_holdout.get("model_utterance_rate", 0.0),
            candidate_dev["near_duplicate_rate"] <= parent_dev["near_duplicate_rate"],
            candidate_holdout["near_duplicate_rate"] <= parent_holdout["near_duplicate_rate"],
            candidate_dev.get("dialogue_near_duplicate_rate", 0.0)
            <= parent_dev.get("dialogue_near_duplicate_rate", 0.0),
            candidate_holdout.get("dialogue_near_duplicate_rate", 0.0)
            <= parent_holdout.get("dialogue_near_duplicate_rate", 0.0),
            candidate_dev.get("mechanical_utterance_rate", 0.0)
            <= parent_dev.get("mechanical_utterance_rate", 0.0),
            candidate_holdout.get("mechanical_utterance_rate", 0.0)
            <= parent_holdout.get("mechanical_utterance_rate", 0.0),
        )
    )
    eligible = (
        holdout_distinct
        and candidate_dev["hard_gate_pass"]
        and candidate_holdout["hard_gate_pass"]
        and non_regression
        and dev_gain >= 1.0
        and holdout_gain >= 1.0
    )
    if not holdout_distinct:
        stop_reason = "holdout must use a different frozen ledger"
    elif not eligible:
        stop_reason = "candidate gain did not pass every dev and holdout gate"
    elif round_no >= max_rounds:
        stop_reason = "maximum RSI rounds reached"
    else:
        stop_reason = "human review required before another shadow round"
    return {
        "schema_version": 1,
        "status": "research_shadow_candidate" if eligible else "parent_retained",
        "round": round_no,
        "max_rounds": max_rounds,
        "dev_gain": round(dev_gain, 2),
        "holdout_gain": round(holdout_gain, 2),
        "non_regression": non_regression,
        "holdout_distinct": holdout_distinct,
        "parent_replacement_allowed": False,
        "promotion_allowed": False,
        "requires_human_approval": True,
        "continue_allowed": eligible and round_no < max_rounds,
        "stop_reason": stop_reason,
        "metrics": {
            "parent_dev": parent_dev,
            "candidate_dev": candidate_dev,
            "parent_holdout": parent_holdout,
            "candidate_holdout": candidate_holdout,
        },
    }


def run_rsi_shadow(args: argparse.Namespace) -> int:
    if args.round < 1 or args.max_rounds < 1 or args.round > args.max_rounds:
        raise ValueError("RSI round must be between 1 and max_rounds")
    paths = {
        "parent_dev": Path(args.parent_dev),
        "candidate_dev": Path(args.candidate_dev),
        "parent_holdout": Path(args.parent_holdout),
        "candidate_holdout": Path(args.candidate_holdout),
    }
    runs = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    if any(run.get("schema_version", 0) < 2 for run in runs.values()):
        raise ValueError("RSI requires event-debate schema_version 2 runs with complete raw logs")

    def ledger_sha(run: dict) -> str:
        recorded = run.get("ledger_sha256")
        if isinstance(recorded, str) and re.fullmatch(r"[0-9a-f]{64}", recorded):
            return recorded
        ledger_path = Path(str(run.get("ledger", "")))
        if not ledger_path.is_file():
            raise ValueError("RSI inputs must reference readable ledger files or include ledger_sha256")
        return hashlib.sha256(ledger_path.read_bytes()).hexdigest()

    ledger_hashes = {name: ledger_sha(run) for name, run in runs.items()}
    for parent_name, candidate_name in (("parent_dev", "candidate_dev"), ("parent_holdout", "candidate_holdout")):
        parent, candidate = runs[parent_name], runs[candidate_name]
        for field in ("model", "domain"):
            if parent.get(field) != candidate.get(field):
                raise ValueError(f"{parent_name}/{candidate_name}: {field} must match")
        if ledger_hashes[parent_name] != ledger_hashes[candidate_name]:
            raise ValueError(f"{parent_name}/{candidate_name}: ledger content must match")
    holdout_distinct = ledger_hashes["parent_dev"] != ledger_hashes["parent_holdout"]
    metrics = {name: event_run_metrics(run) for name, run in runs.items()}
    decision = decide_rsi_shadow(
        metrics["parent_dev"],
        metrics["candidate_dev"],
        metrics["parent_holdout"],
        metrics["candidate_holdout"],
        round_no=args.round,
        max_rounds=args.max_rounds,
        holdout_distinct=holdout_distinct,
    )
    decision["inputs"] = {
        name: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prompt_profile": runs[name].get("prompt_profile", "unknown"),
        }
        for name, path in paths.items()
    }
    write_json(Path(args.out), decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


def run_event_debate(args: argparse.Namespace) -> int:
    runtime_started = time.perf_counter()
    ledger = load_claim_ledger(Path(args.ledger))
    domains = load_domains()
    domain_personas = domains[args.domain]["personas"]
    preferences = ledger.get("role_preferences", {})
    known_personas = {persona["id"] for persona in domain_personas}
    unknown_personas = set(preferences) - known_personas
    if unknown_personas:
        raise ValueError(f"ledger has unknown personas: {sorted(unknown_personas)}")
    personas = [persona for persona in domain_personas if persona["id"] in preferences]
    if len(personas) < 2:
        raise ValueError("event-debate requires at least two active personas")
    if args.max_turns < 1 or args.reconcile_rounds < 0 or args.max_tokens < 1:
        raise ValueError("max-turns/max-tokens must be positive and reconcile-rounds cannot be negative")
    execution = event_execution_settings(args, len(personas))
    if args.backend == "mlx" and not args.model_path:
        raise ValueError("MLX backend requires --model-path")
    shared_renderer_adapter = getattr(args, "renderer_adapter", None)
    body_renderer_adapter = getattr(args, "body_adapter", None)
    no_renderer = bool(getattr(args, "no_renderer", False))
    if sum(bool(value) for value in (args.adapter_map, shared_renderer_adapter, body_renderer_adapter)) > 1:
        raise ValueError("--adapter-map, --renderer-adapter, and --body-adapter are mutually exclusive")
    if no_renderer and (args.adapter_map or shared_renderer_adapter or body_renderer_adapter):
        raise ValueError("--no-renderer cannot be combined with a renderer adapter")
    if args.backend == "ollama" and (args.adapter_map or shared_renderer_adapter or body_renderer_adapter):
        raise ValueError("Ollama backend does not accept MLX renderer adapters")
    for adapter_name, adapter_path in (
        ("renderer", shared_renderer_adapter),
        ("body renderer", body_renderer_adapter),
    ):
        if not adapter_path:
            continue
        adapter_dir = Path(adapter_path)
        if not adapter_dir.is_dir():
            raise ValueError(f"{adapter_name} adapter directory does not exist: {adapter_path}")
        for required in ("adapter_config.json", "adapters.safetensors"):
            if not (adapter_dir / required).is_file():
                raise ValueError(f"{adapter_name} adapter is missing {required}")
    adapter_map = (
        load_adapter_map(args.adapter_map, {persona["id"] for persona in personas})
        if args.backend == "mlx"
        else {}
    )
    catalog = {item["code"]: item for item in ledger["claim_catalog"]}
    data_view = [{"id": item["id"], "text": item["text"]} for item in ledger["data"]]
    catalog_view = [
        {"code": item["code"], "label": item["label"], "supported_by": item["supported_by"]}
        for item in ledger["claim_catalog"]
    ]
    model_id = args.model_path if args.backend == "mlx" else args.ollama_model
    if args.backend == "mlx":
        try:
            import mlx.core as mx
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.tuner.utils import load_adapters, remove_lora_layers
        except ImportError as exc:
            raise ValueError("MLX backendはMLX-LM環境のPythonで実行してください") from exc
        print(f"[MLX] {model_id} をロード", flush=True)
        model, tokenizer = load(model_id)
        model.eval()
        mx.random.seed(args.seed)
        sampler = make_sampler(temp=args.temperature, top_p=0.85)
        body_sampler = make_sampler(temp=0.0)
        active_adapter: str | None = None
        adapter_layers_active = False

        def activate_persona_adapter(persona_id: str | None) -> str | None:
            nonlocal model, active_adapter, adapter_layers_active
            if persona_id == "shared":
                target = shared_renderer_adapter
            elif persona_id == "body":
                target = body_renderer_adapter
            else:
                target = adapter_map.get(persona_id) if persona_id is not None else None
            if target == active_adapter and (target is not None or not adapter_layers_active):
                return target
            if adapter_layers_active:
                model = remove_lora_layers(model)
                adapter_layers_active = False
            if target is not None:
                model = load_adapters(model, target)
                model.eval()
                adapter_layers_active = True
            active_adapter = target
            return target

        def backend_ask_json(
            system: str, user: str, max_tokens: int, deterministic: bool = False
        ) -> tuple[str, dict | None]:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            raw = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=body_sampler if deterministic else sampler,
                verbose=False,
            ).strip()
            return raw, parse_json_object(raw)

    else:
        print(f"[Ollama] {model_id} を使用", flush=True)
        ollama_call_index = 0

        def activate_persona_adapter(persona_id: str | None) -> str | None:
            return None

        def backend_ask_json(
            system: str, user: str, max_tokens: int, deterministic: bool = False
        ) -> tuple[str, dict | None]:
            nonlocal ollama_call_index
            ollama_call_index += 1
            parsed, meta = ask_ollama(
                model=model_id,
                system=system,
                user=user,
                schema="json",
                api_url=args.api_url,
                timeout=args.timeout,
                num_predict=max_tokens,
                temperature=0.0 if deterministic else args.temperature,
                seed=args.seed + ollama_call_index,
                include_raw=True,
            )
            raw = str(meta.pop("_raw_content"))
            return raw, parsed

    model_calls: list[dict] = []

    def ask_json(
        system: str, user: str, max_tokens: int, phase: str, deterministic: bool = False
    ) -> tuple[str, dict | None]:
        started = time.perf_counter()
        raw, parsed = backend_ask_json(system, user, max_tokens, deterministic)
        elapsed = time.perf_counter() - started
        model_calls.append(
            {"index": len(model_calls) + 1, "phase": phase, "seconds": round(elapsed, 3), "max_tokens": max_tokens}
        )
        print(f"[timing] {phase}: {elapsed:.1f}s (call {len(model_calls)})", flush=True)
        return raw, parsed

    created_at = dt.datetime.now().astimezone()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    run_stem = f"event_debate_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
    partial_path = outdir / f"{run_stem}.partial.json"
    final_path = outdir / f"{run_stem}.json"
    run = {
        "schema_version": 2,
        "created_at": created_at.isoformat(timespec="seconds"),
        "model": model_id,
        "backend": args.backend,
        "ledger": args.ledger,
        "ledger_sha256": hashlib.sha256(Path(args.ledger).read_bytes()).hexdigest(),
        "domain": args.domain,
        "prompt_profile": args.prompt_profile,
        "seed": args.seed,
        "execution": {
            "fast": bool(getattr(args, "fast", False)),
            "no_renderer": no_renderer,
            "body_temperature": 0.0 if body_renderer_adapter else None,
            **execution,
        },
        "adapter_map": adapter_map,
        "renderer_adapter": shared_renderer_adapter,
        "body_adapter": body_renderer_adapter,
        "renderer_schema_version": 3,
        "body_renderer_schema_version": 2 if body_renderer_adapter else None,
        "adapter_scope": "claim_body_v2_only" if body_renderer_adapter else "utterance_renderer_v3_only",
        "adapter_fingerprints": {
            persona_id: {
                "config_sha256": hashlib.sha256((Path(path) / "adapter_config.json").read_bytes()).hexdigest(),
                "weights_sha256": hashlib.sha256((Path(path) / "adapters.safetensors").read_bytes()).hexdigest(),
            }
            for persona_id, path in adapter_map.items()
        },
        "renderer_adapter_fingerprint": (
            {
                "config_sha256": hashlib.sha256(
                    (Path(shared_renderer_adapter) / "adapter_config.json").read_bytes()
                ).hexdigest(),
                "weights_sha256": hashlib.sha256(
                    (Path(shared_renderer_adapter) / "adapters.safetensors").read_bytes()
                ).hexdigest(),
            }
            if shared_renderer_adapter
            else None
        ),
        "body_adapter_fingerprint": (
            {
                "config_sha256": hashlib.sha256(
                    (Path(body_renderer_adapter) / "adapter_config.json").read_bytes()
                ).hexdigest(),
                "weights_sha256": hashlib.sha256(
                    (Path(body_renderer_adapter) / "adapters.safetensors").read_bytes()
                ).hexdigest(),
            }
            if body_renderer_adapter
            else None
        ),
        "independent": {},
        "events": [],
        "reconciliation": [],
        "renderer_batches": [],
    }
    persona_configs = {persona["id"]: persona for persona in personas}
    persona_names = {persona["id"]: persona["name"] for persona in personas}

    def render_utterance_records(records: list[dict], phase: str) -> None:
        if not records:
            return
        if body_renderer_adapter:
            for record_index, record in enumerate(records):
                adapter_path = activate_persona_adapter("body")
                batch_number = len(run["renderer_batches"]) + 1
                batch_id = f"RB{batch_number:02d}"
                renderer_id = "B01"
                item = {
                    "id": renderer_id,
                    "speaker": persona_configs[record["persona_id"]]["name"],
                    "claim": record["label"],
                }
                raw, parsed = ask_json(
                    BODY_RENDERER_SYSTEM,
                    json.dumps({"items": [item]}, ensure_ascii=False),
                    min(args.max_tokens, 180),
                    f"body-renderer:{phase}:{record['id']}",
                    deterministic=True,
                )
                values, schema_warning, schema_repaired = parse_renderer_bodies(
                    parsed, [renderer_id]
                )
                body, body_warning = normalize_renderer_body(values.get(renderer_id))
                if body is not None and not body_is_neutral(body):
                    body, body_warning = None, "body renderer exposed a dialogue move"
                if body is not None and not body_is_polite_sentence(body):
                    body, body_warning = None, "body renderer did not return a polite complete sentence"
                if body is not None and not body_matches_claim(body, record["label"]):
                    body, body_warning = None, "body renderer does not match the frozen claim"
                if body is not None and not dialogue_numbers_are_grounded(body, item):
                    body, body_warning = None, "body renderer invents an ungrounded number"
                if body is not None and dialogue_selects_competing_claim(
                    body, record.get("competitor_labels", [])
                ):
                    body, body_warning = None, "body renderer selects a competing frozen claim"
                utterance = (
                    compose_dialogue_body(
                        body,
                        record["label"],
                        record.get("validation_move") or "propose",
                        record_index,
                    )
                    if body is not None
                    else None
                )
                target = record["target"]
                if utterance is None:
                    utterance = record["fallback"]
                    origin = record.get("fallback_origin", "statement_fallback")
                else:
                    origin = "model_body_v2_schema_repair" if schema_repaired else "model_body_v2"
                run["renderer_batches"].append(
                    {
                        "batch_id": batch_id,
                        "record_id": record["id"],
                        "phase": phase,
                        "renderer_kind": "claim_body_v2",
                        "persona_id": record["persona_id"],
                        "adapter": adapter_path,
                        "request": {"items": [item]},
                        "raw": raw,
                        "warning": body_warning or schema_warning,
                    }
                )
                target["utterance"] = utterance
                target["utterance_origin"] = origin
                target["utterance_warning"] = body_warning or schema_warning
                target["renderer_batch_id"] = batch_id
                target["renderer_adapter"] = adapter_path
            return
        if no_renderer or (
            getattr(args, "fast", False)
            and not adapter_map
            and not shared_renderer_adapter
            and not body_renderer_adapter
        ):
            print(
                f"[{'no-renderer' if no_renderer else 'fast'}] {phase} rendererを省略し、"
                "検証済みstatementを表示します。",
                flush=True,
            )
            for record in records:
                target = record["target"]
                target["utterance"] = record["fallback"]
                target["utterance_origin"] = record.get("fallback_origin", "statement_fallback")
                target["utterance_warning"] = "fast mode skipped the model renderer"
                target["renderer_batch_id"] = None
                target["renderer_adapter"] = None
            return
        grouped: dict[tuple[str, int], list[dict]] = {}
        group_counts: dict[str, int] = {}
        batch_limit = 1 if shared_renderer_adapter or adapter_map else 3
        for record in records:
            base = record["persona_id"] if adapter_map else "shared"
            key = (base, group_counts.get(base, 0) // batch_limit)
            grouped.setdefault(key, []).append(record)
            group_counts[base] = group_counts.get(base, 0) + 1
        for (group_key, chunk_index), batch in grouped.items():
            persona = persona_configs[group_key] if adapter_map else None
            adapter_path = activate_persona_adapter(
                group_key if adapter_map else "shared" if shared_renderer_adapter else None
            )
            items = []
            for record in batch:
                item = {"id": record["id"], **record["payload"]}
                item["speech_act"] = renderer_move_instruction(item.get("move"))
                if persona is None:
                    source = persona_configs[record["persona_id"]]
                    item["speaker"] = source["name"]
                items.append(item)
            raw, parsed = ask_json(
                renderer_system(persona),
                json.dumps({"items": items}, ensure_ascii=False),
                min(args.max_tokens, max(220, 80 + 90 * len(items))),
                f"renderer:{phase}:{group_key}:{chunk_index}",
            )
            values, batch_warning = parse_renderer_utterances(
                parsed, [record["id"] for record in batch]
            )
            batch_id = f"RB{len(run['renderer_batches']) + 1:02d}"
            run["renderer_batches"].append(
                {
                    "batch_id": batch_id,
                    "phase": phase,
                    "persona_id": None if persona is None else group_key,
                    "adapter": adapter_path,
                    "request": {"items": items},
                    "raw": raw,
                    "warning": batch_warning,
                }
            )
            for record in batch:
                target = record["target"]
                candidate = restore_claim_label(values.get(record["id"]), record["label"])
                move = record.get("validation_move")
                if move:
                    utterance, warning = validate_dialogue_move(candidate, move)
                else:
                    utterance, warning = validate_dialogue_utterance(candidate)
                if utterance is not None and not dialogue_is_aligned(
                    utterance, record["label"], record.get("competitor_labels", [])
                ):
                    utterance, warning = None, "renderer utterance does not match the frozen claim"
                if utterance is not None and dialogue_selects_competing_claim(
                    utterance, record.get("competitor_labels", [])
                ):
                    utterance, warning = None, "renderer utterance selects a competing frozen claim"
                if utterance is not None and not dialogue_numbers_are_grounded(
                    utterance, record["payload"]
                ):
                    utterance, warning = None, "renderer utterance invents an ungrounded number"
                target_label = record.get("target_label")
                if (
                    utterance is not None
                    and target_label
                    and move in {"object", "agree"}
                    and not reaction_is_aligned(utterance, record["label"], target_label, move)
                ):
                    utterance, warning = None, "renderer utterance follows the target instead of its own claim"
                sanitized = False
                if utterance is None and candidate is not None:
                    repaired = sanitize_dialogue_move(candidate, move or "")
                    if (
                        repaired is not None
                        and dialogue_is_aligned(
                            repaired, record["label"], record.get("competitor_labels", [])
                        )
                        and not dialogue_selects_competing_claim(
                            repaired, record.get("competitor_labels", [])
                        )
                        and dialogue_numbers_are_grounded(repaired, record["payload"])
                    ):
                        if not target_label or move not in {"object", "agree"} or reaction_is_aligned(
                            repaired, record["label"], target_label, move
                        ):
                            utterance = repaired
                            sanitized = True
                if utterance is None:
                    utterance = record["fallback"]
                    origin = record.get("fallback_origin", "statement_fallback")
                else:
                    origin = "model_renderer_v3_sanitized" if sanitized else "model_renderer_v3"
                target["utterance"] = utterance
                target["utterance_origin"] = origin
                target["utterance_warning"] = batch_warning or warning
                target["renderer_batch_id"] = batch_id
                target["renderer_adapter"] = adapter_path

    write_json(partial_path, run)
    persona_claims: dict[str, list[dict]] = {}
    print("\n######## 独立再計算 / 他者の文章は非公開 ########", flush=True)
    for persona in personas:
        activate_persona_adapter(None)
        adapter_path = adapter_map.get(persona["id"])
        available_codes = [code for code in preferences.get(persona["id"], []) if code in catalog]
        available_catalog = [
            {"code": code, "label": catalog[code]["label"], "supported_by": catalog[code]["supported_by"]}
            for code in available_codes
        ]
        objective = (
            f"最大化する効用: {persona['utility']} 最小化する損失: {persona['loss']}"
            if args.prompt_profile != "baseline"
            else persona["objective"]
        )
        system = (
            f"あなたは{persona['name']}。他人格の文章は一切見ていない。{objective}。"
            "dataとclaim_catalogだけを独立確認する。台帳外の主張は禁止。"
            "statementは内部思考ではなく、選んだlabelとdataを根拠にした公開用の自然文1文とする。"
            "会話文は別rendererが作るため生成しない。"
            "指定JSONだけを返す。"
        )
        prompt_payload = {
            "topic": ledger["topic"],
            "data": data_view,
            "claim_catalog": available_catalog,
            "persona_focus": persona["objective"],
            "rule": (
                f"claim_catalogから重要順に{execution['claims_per_persona']}件返す。JSONはclaims配列のみ。"
                "各要素のキーはcode, data_ids, confidence, statementだけ。ダミー語CODEは禁止。"
                "data_idsは選んだcodeのsupported_byから必要なものを選ぶ。"
                "statementは240字以内の自然な日本語1文で、選んだD番号を[D01]の形で必ず引用し、"
                "labelの意味やdataにない事実を追加しない。"
            ),
        }
        if args.prompt_profile == "orthogonal_fewshot" and available_catalog:
            prompt_payload["format_example"] = {
                "claims": [
                    {
                        "code": example["code"],
                        "data_ids": [example["supported_by"][0]],
                        "confidence": "MEDIUM",
                        "statement": (
                            f"{example['label']}と判断します。"
                            f"根拠は[{example['supported_by'][0]}]です。"
                        ),
                    }
                    for example in available_catalog[: execution["claims_per_persona"]]
                ]
            }
        user = json.dumps(
            prompt_payload,
            ensure_ascii=False,
        )
        raw, parsed = ask_json(
            system,
            user,
            execution["decision_max_tokens"],
            f"independent:{persona['id']}",
        )
        valid, rejected, warnings, seen_codes = [], [], [], set()
        for claim in (parsed or {}).get("claims", []):
            normalized, reason = validate_coded_claim(claim, ledger)
            if normalized and normalized["code"] not in seen_codes:
                normalized["origin"] = "model"
                statement, statement_reason = validate_public_statement(claim.get("statement"), normalized["data_ids"])
                if statement is None:
                    statement = sanitize_model_statement(claim.get("statement"), normalized["data_ids"])
                    if statement is None:
                        statement = label_statement(normalized["code"], normalized["data_ids"], ledger)
                        normalized["statement_origin"] = "label_fallback"
                    else:
                        normalized["statement_origin"] = "model_sanitized"
                    warnings.append({"code": normalized["code"], "reason": statement_reason})
                else:
                    normalized["statement_origin"] = "model"
                normalized["statement"] = statement
                normalized["utterance"] = dialogue_fallback(statement)
                normalized["utterance_origin"] = "statement_fallback"
                valid.append(normalized)
                seen_codes.add(normalized["code"])
            else:
                rejected.append({"claim": claim, "reason": reason or "duplicate code"})
        for code in preferences.get(persona["id"], []):
            if len(valid) >= execution["claims_per_persona"]:
                break
            if code in catalog and code not in seen_codes:
                valid.append(
                    {
                        "code": code,
                        "data_ids": catalog[code]["supported_by"][:2],
                        "confidence": 60,
                        "origin": "validated_fallback",
                        "statement": label_statement(code, catalog[code]["supported_by"][:2], ledger),
                        "statement_origin": "label_fallback",
                        "utterance": dialogue_fallback(
                            label_statement(code, catalog[code]["supported_by"][:2], ledger)
                        ),
                        "utterance_origin": "statement_fallback",
                    }
                )
                seen_codes.add(code)
        dialogue_render_raw = None
        dialogue_render_repair_raw = None
        persona_claims[persona["id"]] = valid
        run["independent"][persona["id"]] = {
            "raw": raw,
            "valid": valid,
            "rejected": rejected,
            "warnings": warnings,
            "adapter": None,
            "renderer_adapter": adapter_path,
            "dialogue_render_raw": dialogue_render_raw,
            "dialogue_render_repair_raw": dialogue_render_repair_raw,
        }
        print(f"\n[{persona['name']}] 採用={len(valid)} 失格={len(rejected)}", flush=True)
        for claim in valid:
            origin = "モデル自然文" if claim["statement_origin"] == "model" else "ラベル補完"
            print(f"  [{origin}/{claim['origin']}] {claim['statement']}", flush=True)
            print(f"    {claim['code']} <- {','.join(claim['data_ids'])}", flush=True)
        for item in rejected:
            print(f"  [失格] {item['reason']}", flush=True)
        for item in warnings:
            print(f"  [自然文補完/{item.get('field', 'statement')}] {item['code']}: {item['reason']}", flush=True)
        write_json(partial_path, run)

    events = schedule_claim_events(persona_claims, ledger, execution["max_turns"])
    run["events"] = events
    write_json(partial_path, run)
    print("\n######## イベント駆動討論 / 異議・賛同を優先 ########", flush=True)
    event_by_id = {event["claim_id"]: event for event in events}
    event_renderer_records = []
    for event_index, event in enumerate(events):
        target_event = event_by_id.get(event["target_claim_id"])
        move = renderer_event_move(event["action"])
        move_fallback, fallback_origin = compose_dialogue_fallback(
            event["statement"], event["label"], move, event_index
        )
        event_renderer_records.append(
            {
                "id": event["claim_id"],
                "persona_id": event["persona_id"],
                "target": event,
                "label": event["label"],
                "target_label": target_event["label"] if target_event else None,
                "validation_move": move,
                "fallback": move_fallback,
                "fallback_origin": fallback_origin,
                "payload": {
                    "phase": "event",
                    "move": move,
                    "own_claim": event["label"],
                    "target_claim": target_event["label"] if target_event else None,
                    "evidence": [item["text"] for item in data_view if item["id"] in event["data_ids"]],
                },
            }
        )
    render_utterance_records(event_renderer_records, "event")
    for event in events:
        event["reaction_data_ids"] = event["data_ids"] if event["action"] in {"object", "agree_extend"} else []
        event["reaction_warning"] = event.get("utterance_warning")
        target = f" -> {event['target_claim_id']}" if event["target_claim_id"] else ""
        print(
            f"{event['claim_id']} [{event['action_label']}]{target} {persona_names[event['persona_id']]}: "
            f"{event['utterance']}",
            flush=True,
        )
        print(
            f"    根拠: {','.join(event['data_ids'])} / {event['utterance_origin']} / {event['origin']}",
            flush=True,
        )
    write_json(partial_path, run)

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
    previous_votes = {}
    persona_rank = {persona["id"]: index for index, persona in enumerate(personas)}
    dialogue_move_rank = {"agree": 0, "maintain": 1, "revise": 2}
    for round_no in range(1, execution["reconcile_rounds"] + 1):
        print(f"\n######## すり合わせ {round_no} / 対立ごとの一問一答 ########", flush=True)
        votes = {}
        for pair in contested_pairs:
            key = "|".join(pair)
            pair_votes = {}
            print(f"論点: {pair[0]} vs {pair[1]}", flush=True)
            for persona in personas:
                activate_persona_adapter(None)
                adapter_path = adapter_map.get(persona["id"])
                system = (
                    f"あなたは{persona['name']}。最大化する効用は「{persona['utility']}」、"
                    f"最小化する損失は「{persona['loss']}」。"
                    "検証済み2コードを比較し、選択と証拠付きstatementを指定JSONで返す。"
                    "会話文は別rendererが作るため生成しない。"
                )
                pair_data_ids = sorted(set(catalog[pair[0]]["supported_by"]) | set(catalog[pair[1]]["supported_by"]))
                prior_vote = previous_votes.get(key, {}).get(persona["id"], {})
                previous_choice = prior_pair_choice(persona_claims[persona["id"]], pair, prior_vote)
                user = json.dumps(
                    {
                        "left": {
                            "code": pair[0],
                            "label": catalog[pair[0]]["label"],
                            "supported_by": catalog[pair[0]]["supported_by"],
                        },
                        "right": {
                            "code": pair[1],
                            "label": catalog[pair[1]]["label"],
                            "supported_by": catalog[pair[1]]["supported_by"],
                        },
                        "data": [item for item in data_view if item["id"] in pair_data_ids],
                        "own_initial_codes": [claim["code"] for claim in persona_claims[persona["id"]]],
                        "own_previous_choice": previous_choice,
                        "previous_tally": previous_tally.get(key, {}),
                        "rule": (
                            f"choiceは {pair[0]} または {pair[1]} または BOTH または ABSTAIN。"
                            "data_idsは表示したdataから1〜2件。statementは240字以内の自然な日本語1文で、"
                            "選択したcodeのsupported_byにあるD番号だけを[D01]形式で引用する。"
                            "change_reasonは前ラウンドから選択を変えた時だけ、その理由とD番号を書く。"
                            "変えていない時は空文字。"
                            "JSONキーはchoice,data_ids,statement,change_reasonだけ。"
                        ),
                        "format_example": {
                            "choice": pair[0],
                            "data_ids": [catalog[pair[0]]["supported_by"][0]],
                            "statement": (
                                f"{catalog[pair[0]]['label']}を採ります。"
                                f"根拠は[{catalog[pair[0]]['supported_by'][0]}]です。"
                            ),
                            "change_reason": (
                                f"前回の選択より[{catalog[pair[0]]['supported_by'][0]}]を重視して変更しました。"
                                if previous_choice and previous_choice != pair[0]
                                else ""
                            ),
                        }
                        if args.prompt_profile == "orthogonal_fewshot"
                        else None,
                    },
                    ensure_ascii=False,
                )
                raw, parsed = ask_json(
                    system,
                    user,
                    min(execution["decision_max_tokens"], 300),
                    f"reconciliation:{round_no}:{key}:{persona['id']}",
                )
                allowed_choices = {*pair, "BOTH", "ABSTAIN"}
                choice = parsed.get("choice") if isinstance(parsed, dict) else None
                choice_origin = "model_json"
                if choice not in allowed_choices:
                    matches = [code for code in pair if code in raw]
                    if len(matches) == 1:
                        choice = matches[0]
                    elif "BOTH" in raw or len(matches) == 2:
                        choice = "BOTH"
                    else:
                        choice = "ABSTAIN"
                    choice_origin = "parse_fallback"
                allowed_data = (
                    set(catalog[choice]["supported_by"])
                    if choice in catalog
                    else set(pair_data_ids)
                )
                proposed_data = parsed.get("data_ids") if isinstance(parsed, dict) else None
                if (
                    not isinstance(proposed_data, list)
                    or not proposed_data
                    or not all(isinstance(value, str) for value in proposed_data)
                    or not set(proposed_data).issubset(allowed_data)
                ):
                    data_ids = sorted(allowed_data)[:1]
                else:
                    data_ids = list(dict.fromkeys(proposed_data))[:2]
                statement, statement_reason = validate_public_statement(
                    parsed.get("statement") if isinstance(parsed, dict) else None,
                    data_ids,
                )
                if statement is None:
                    if choice in catalog:
                        statement = label_statement(choice, data_ids, ledger)
                    elif choice == "BOTH":
                        statement = f"両方を独立シナリオとして残します。根拠は[{','.join(data_ids)}]。"
                    else:
                        statement = f"証拠だけでは選べないため保留します。参照は[{','.join(data_ids)}]。"
                    statement_origin = "label_fallback"
                else:
                    statement_origin = "model"
                changed = previous_choice is not None and previous_choice != choice
                dialogue_move = (
                    "revise"
                    if changed
                    else "maintain" if previous_choice == choice else "agree"
                )
                selected_label = (
                    catalog[choice]["label"]
                    if choice in catalog
                    else "両方を残す" if choice == "BOTH" else "判断を保留する"
                )
                utterance = dialogue_fallback(statement)
                utterance_origin = "statement_fallback"
                utterance_warning = None
                change_reason = ""
                change_reason_origin = "not_required"
                change_reason_warning = None
                if changed:
                    change_reason, change_reason_warning = validate_public_statement(
                        parsed.get("change_reason") if isinstance(parsed, dict) else None,
                        data_ids,
                    )
                    if change_reason is None:
                        change_reason = (
                            f"前回の{previous_choice}から{choice}へ変更しました。"
                            f"新しい根拠は[{','.join(data_ids)}]です。"
                        )
                        change_reason_origin = "label_fallback"
                    else:
                        change_reason_origin = "model"
                repair_raw = None
                repair_statement_warning = None
                repair_change_reason_warning = None
                repair_utterance_warning = None
                if (
                    statement_reason is not None
                    or (changed and change_reason_warning is not None)
                ):
                    repair_label = selected_label
                    repair_system = (
                        f"あなたは{persona['name']}。選択は{choice}で確定済み。再評価は禁止。"
                        "渡されたD番号だけで公開用自然文を修復し、指定JSONだけを返す。"
                    )
                    repair_user = json.dumps(
                        {
                            "choice": choice,
                            "label": repair_label,
                            "allowed_data_ids": data_ids,
                            "data": [item for item in data_view if item["id"] in data_ids],
                            "previous_choice": previous_choice,
                            "changed": changed,
                            "dialogue_move": dialogue_move,
                            "rule": (
                                "statementはlabelを自然な日本語1文にし、allowed_data_idsだけを[D01]形式で引用する。"
                                "change_reasonはchanged=trueの時だけ、前回から変えた理由とallowed_data_idsを引用する。"
                                "JSONキーはstatement,change_reasonだけ。"
                            ),
                            "format_example": {
                                "statement": f"{repair_label}と判断します。根拠は[{data_ids[0]}]です。",
                                "change_reason": (
                                    f"前回より[{data_ids[0]}]を重視して選択を変更しました。" if changed else ""
                                ),
                            },
                        },
                        ensure_ascii=False,
                    )
                    repair_raw, repair_parsed = ask_json(
                        repair_system,
                        repair_user,
                        min(execution["decision_max_tokens"], 240),
                        f"reconciliation-repair:{round_no}:{key}:{persona['id']}",
                    )
                    if statement_reason is not None:
                        repaired, repair_statement_warning = validate_public_statement(
                            repair_parsed.get("statement") if isinstance(repair_parsed, dict) else None,
                            data_ids,
                        )
                        if repaired is not None:
                            statement = repaired
                            statement_origin = "model_repair"
                        else:
                            sanitized = sanitize_model_statement(
                                repair_parsed.get("statement") if isinstance(repair_parsed, dict) else None,
                                data_ids,
                            )
                            if sanitized is not None:
                                statement = sanitized
                                statement_origin = "model_sanitized"
                    if changed and change_reason_warning is not None:
                        repaired_reason, repair_change_reason_warning = validate_public_statement(
                            repair_parsed.get("change_reason") if isinstance(repair_parsed, dict) else None,
                            data_ids,
                        )
                        if repaired_reason is not None:
                            change_reason = repaired_reason
                            change_reason_origin = "model_repair"
                        else:
                            sanitized_reason = sanitize_model_statement(
                                repair_parsed.get("change_reason") if isinstance(repair_parsed, dict) else None,
                                data_ids,
                            )
                            if sanitized_reason is not None:
                                change_reason = sanitized_reason
                                change_reason_origin = "model_sanitized"
                vote = {
                    "choice": choice,
                    "choice_origin": choice_origin,
                    "data_ids": data_ids,
                    "statement": statement,
                    "statement_origin": statement_origin,
                    "statement_warning": statement_reason,
                    "utterance": utterance,
                    "utterance_origin": utterance_origin,
                    "utterance_warning": utterance_warning,
                    "dialogue_move": dialogue_move,
                    "changed_from_previous": changed,
                    "previous_choice": previous_choice,
                    "change_reason": change_reason,
                    "change_reason_origin": change_reason_origin,
                    "change_reason_warning": change_reason_warning,
                    "raw": raw,
                    "repair_raw": repair_raw,
                    "repair_statement_warning": repair_statement_warning,
                    "repair_utterance_warning": repair_utterance_warning,
                    "repair_change_reason_warning": repair_change_reason_warning,
                    "adapter": None,
                    "renderer_adapter": adapter_path,
                }
                pair_votes[persona["id"]] = vote
            votes[key] = pair_votes
            reconciliation_renderer_records = []
            for persona_id, vote in pair_votes.items():
                choice = vote["choice"]
                selected_label = (
                    catalog[choice]["label"]
                    if choice in catalog
                    else "両方を残す" if choice == "BOTH" else "判断を保留する"
                )
                vote_fallback, vote_fallback_origin = compose_dialogue_fallback(
                    vote["statement"],
                    selected_label,
                    vote["dialogue_move"],
                    persona_rank[persona_id] + round_no,
                )
                reconciliation_renderer_records.append(
                    {
                        "id": f"R{round_no}:{key}:{persona_id}",
                        "persona_id": persona_id,
                        "target": vote,
                        "label": selected_label,
                        "target_label": None,
                        "competitor_labels": [
                            catalog[code]["label"]
                            for code in pair
                            if catalog[code]["label"] != selected_label
                        ],
                        "validation_move": vote["dialogue_move"],
                        "fallback": vote_fallback,
                        "fallback_origin": vote_fallback_origin,
                        "payload": {
                            "phase": "reconciliation",
                            "move": vote["dialogue_move"],
                            "previous_choice": vote["previous_choice"],
                            "selected_claim": selected_label,
                            "alternatives": [catalog[code]["label"] for code in pair],
                            "evidence": [
                                item["text"] for item in data_view if item["id"] in vote["data_ids"]
                            ],
                        },
                    }
                )
            render_utterance_records(reconciliation_renderer_records, f"reconciliation-{round_no}")
            for persona_id, vote in sorted(
                pair_votes.items(),
                key=lambda item: (dialogue_move_rank[item[1]["dialogue_move"]], persona_rank[item[0]]),
            ):
                print(
                    f"  {persona_names[persona_id]}: {vote['utterance']} "
                    f"({vote['dialogue_move']}/{vote['utterance_origin']})",
                    flush=True,
                )
                print(
                    f"    選択: {vote['choice']} / 根拠: {','.join(vote['data_ids'])}",
                    flush=True,
                )
                if vote["changed_from_previous"]:
                    print(
                        f"    変更理由: {vote['change_reason']} ({vote['change_reason_origin']})",
                        flush=True,
                    )
        previous_tally = {
            key: {
                choice: [vote["choice"] for vote in pair_votes.values()].count(choice)
                for choice in {vote["choice"] for vote in pair_votes.values()}
            }
            for key, pair_votes in votes.items()
        }
        run["reconciliation"].append({"round": round_no, "votes": votes, "tally": previous_tally})
        previous_votes = votes
        write_json(partial_path, run)
        if reconciliation_has_supermajority(previous_tally, contested_pairs, len(personas)):
            print("3/4以上で対立が解決したため、すり合わせを終了します。", flush=True)
            break

    summary = synthesize_event_summary(events, catalog, run["reconciliation"], len(personas))
    run["summary"] = summary
    run["runtime"] = {
        "model_call_count": len(model_calls),
        "model_seconds": round(sum(call["seconds"] for call in model_calls), 3),
        "elapsed_seconds": round(time.perf_counter() - runtime_started, 3),
        "calls": model_calls,
    }
    run["metrics"] = event_run_metrics(run)
    print("\n######## 検証済み統合 ########", flush=True)
    print("複数人格が合意:", ", ".join(summary["consensus"]) or "なし", flush=True)
    print("異議なしの検証済み主張:", ", ".join(summary["unopposed_supported"]) or "なし", flush=True)
    print("解決した対立:", summary["resolved_conflicts"] or "なし", flush=True)
    print("対立継続:", summary["unresolved_conflicts"] or "なし", flush=True)
    print(
        f"実行時間: {run['runtime']['elapsed_seconds']:.1f}s / model call {len(model_calls)}回",
        flush=True,
    )
    print("RSI shadow指標:", json.dumps(run["metrics"], ensure_ascii=False), flush=True)
    write_json(partial_path, run)
    partial_path.replace(final_path)
    print(f"保存: {final_path}", flush=True)
    return 0


def event_review_sha256(run: dict) -> str:
    payload = {key: value for key, value in run.items() if key != "training_review"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def mark_event_run(args: argparse.Namespace) -> int:
    path = Path(args.run)
    if path.name.endswith(".partial.json"):
        raise ValueError("partial event run cannot be reviewed")
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("schema_version", 0) < 2 or not isinstance(run.get("events"), list):
        raise ValueError(f"event-debate runではありません: {path}")
    metrics = event_run_metrics(run)
    run["metrics"] = metrics
    if args.status == "approved" and not metrics["hard_gate_pass"]:
        raise ValueError("hard gateを通過していないevent runは承認できません")
    run["training_review"] = {
        "status": args.status,
        "reviewed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "reviewer": args.reviewer,
        "note": args.note,
        "content_sha256": event_review_sha256(run),
    }
    write_json(path, run)
    print(
        f"{path}: {args.status} sha256={run['training_review']['content_sha256']}",
        flush=True,
    )
    return 0


def export_dialogue_sft(args: argparse.Namespace) -> int:
    if args.min_per_persona < 1:
        raise ValueError("min-per-persona must be positive")
    batch_size = getattr(args, "batch_size", 1)
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    domains = load_domains()
    grouped: dict[tuple[str, str], list[dict]] = {}
    frozen_splits: dict[tuple[str, str], dict[str, list[dict]]] = {}
    seen: dict[tuple[str, str], list[str]] = {}
    sources: dict[tuple[str, str], set[str]] = {}
    excluded = {"unapproved": 0, "fallback": 0, "mechanical": 0, "duplicate": 0, "invalid": 0}

    def add_example(
        *,
        domain: str,
        persona_id: str,
        system: str,
        user_payload: dict,
        utterance: object,
        origin: object,
        source_sha: str,
    ) -> None:
        key = (domain, persona_id)
        if origin not in MODEL_UTTERANCE_ORIGINS:
            excluded["fallback"] += 1
            return
        normalized, _ = validate_dialogue_utterance(utterance)
        if normalized is None:
            excluded["invalid"] += 1
            return
        if is_mechanical_utterance(normalized):
            excluded["mechanical"] += 1
            return
        if any(similarity(normalized, previous) >= 0.9 for previous in seen.setdefault(key, [])):
            excluded["duplicate"] += 1
            return
        seen[key].append(normalized)
        grouped.setdefault(key, []).append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    {
                        "role": "assistant",
                        "content": json.dumps({"utterance": normalized}, ensure_ascii=False),
                    },
                ]
            }
        )
        sources.setdefault(key, set()).add(source_sha)

    for run_name in args.runs:
        path = Path(run_name)
        run = json.loads(path.read_text(encoding="utf-8"))
        review = run.get("training_review", {})
        if review.get("status") != "approved":
            excluded["unapproved"] += 1
            continue
        if review.get("content_sha256") != event_review_sha256(run):
            raise ValueError(f"review後にevent runが変更されています: {path}")
        if not event_run_metrics(run)["hard_gate_pass"]:
            raise ValueError(f"hard gate再計算に失敗しました: {path}")
        domain = str(run.get("domain"))
        if domain not in domains:
            raise ValueError(f"unknown domain in event run: {domain}")
        ledger_path = Path(str(run.get("ledger")))
        ledger = load_claim_ledger(ledger_path)
        if hashlib.sha256(ledger_path.read_bytes()).hexdigest() != run.get("ledger_sha256"):
            raise ValueError(f"ledgerがevent run作成後に変更されています: {ledger_path}")
        catalog = {item["code"]: item for item in ledger["claim_catalog"]}
        data = {item["id"]: item["text"] for item in ledger["data"]}
        personas = {persona["id"]: persona for persona in domains[domain]["personas"]}
        events = {event["claim_id"]: event for event in run["events"]}
        source_sha = str(review["content_sha256"])
        for event in run["events"]:
            persona_id = event.get("persona_id")
            if persona_id not in personas or event.get("code") not in catalog:
                excluded["invalid"] += 1
                continue
            persona = personas[persona_id]
            target = events.get(event.get("target_claim_id"))
            system = (
                f"あなたは{persona['name']}。最大化する効用: {persona['utility']}。"
                f"最小化する損失: {persona['loss']}。構造化主張から会話文だけを返す。"
            )
            add_example(
                domain=domain,
                persona_id=persona_id,
                system=system,
                user_payload={
                    "phase": "event",
                    "move": event.get("action"),
                    "own_claim": catalog[event["code"]]["label"],
                    "target_claim": catalog[target["code"]]["label"] if target else None,
                    "evidence": [data[value] for value in event.get("data_ids", []) if value in data],
                },
                utterance=event.get("utterance"),
                origin=event.get("utterance_origin"),
                source_sha=source_sha,
            )
        for round_data in run.get("reconciliation", []):
            for pair_key, votes in (round_data.get("votes") or {}).items():
                pair = pair_key.split("|")
                for persona_id, vote in votes.items():
                    choice = vote.get("choice") if isinstance(vote, dict) else None
                    if persona_id not in personas or choice not in catalog:
                        excluded["invalid"] += 1
                        continue
                    persona = personas[persona_id]
                    system = (
                        f"あなたは{persona['name']}。最大化する効用: {persona['utility']}。"
                        f"最小化する損失: {persona['loss']}。すり合わせの会話文だけを返す。"
                    )
                    add_example(
                        domain=domain,
                        persona_id=persona_id,
                        system=system,
                        user_payload={
                            "phase": "reconciliation",
                            "move": vote.get("dialogue_move"),
                            "previous_choice": vote.get("previous_choice"),
                            "selected_claim": catalog[choice]["label"],
                            "alternatives": [catalog[code]["label"] for code in pair if code in catalog],
                            "evidence": [data[value] for value in vote.get("data_ids", []) if value in data],
                        },
                        utterance=vote.get("utterance"),
                        origin=vote.get("utterance_origin"),
                        source_sha=source_sha,
                    )

    out = Path(args.out)
    manifests = {}
    for (domain, persona_id), examples in sorted(grouped.items()):
        target = out / domain / persona_id
        splits = split_examples(examples)
        frozen_splits[(domain, persona_id)] = splits
        utterance_counts = {name: len(rows) for name, rows in splits.items()}
        if batch_size > 1:
            persona = next(item for item in domains[domain]["personas"] if item["id"] == persona_id)
            splits = {
                name: batch_renderer_examples(rows, persona, batch_size)
                for name, rows in splits.items()
            }
        for name, rows in splits.items():
            write_jsonl(target / f"{name}.jsonl", rows)
        ready = (
            len(examples) >= args.min_per_persona
            and bool(splits["valid"])
            and bool(splits["test"])
        )
        manifest = {
            "domain": domain,
            "persona_id": persona_id,
            "counts": {name: len(rows) for name, rows in splits.items()},
            "utterance_counts": utterance_counts,
            "total": len(examples),
            "minimum": args.min_per_persona,
            "renderer_schema": 3 if batch_size > 1 else 2,
            "batch_size": batch_size,
            "source_run_sha256": sorted(sources[(domain, persona_id)]),
            "ready_for_training": ready,
        }
        write_json(target / "manifest.json", manifest)
        manifests[f"{domain}/{persona_id}"] = manifest
        print(f"{domain}/{persona_id}: {len(examples)}/{args.min_per_persona} ready={ready}")
    if getattr(args, "shared_renderer", False):
        for domain in sorted({key[0] for key in frozen_splits}):
            shared_splits = {}
            shared_utterance_counts = {}
            for split_name in ("train", "valid", "test"):
                persona_rows = []
                for persona_id in sorted(key[1] for key in frozen_splits if key[0] == domain):
                    persona = next(
                        item for item in domains[domain]["personas"] if item["id"] == persona_id
                    )
                    persona_rows.append(
                        [(persona, row) for row in frozen_splits[(domain, persona_id)][split_name]]
                    )
                interleaved = [
                    row
                    for group in itertools.zip_longest(*persona_rows)
                    for row in group
                    if row is not None
                ]
                shared_utterance_counts[split_name] = len(interleaved)
                shared_splits[split_name] = batch_shared_renderer_examples(interleaved, batch_size)
            target = out / domain / "shared_renderer"
            for split_name, rows in shared_splits.items():
                write_jsonl(target / f"{split_name}.jsonl", rows)
            shared_manifest = {
                "domain": domain,
                "persona_id": "shared_renderer",
                "counts": {name: len(rows) for name, rows in shared_splits.items()},
                "utterance_counts": shared_utterance_counts,
                "total": sum(shared_utterance_counts.values()),
                "renderer_schema": 3,
                "batch_size": batch_size,
                "ready_for_training": all(shared_splits.values()),
            }
            write_json(target / "manifest.json", shared_manifest)
            manifests[f"{domain}/shared_renderer"] = shared_manifest
            print(
                f"{domain}/shared_renderer: {shared_manifest['total']} utterances / "
                f"{sum(shared_manifest['counts'].values())} batches",
                flush=True,
            )
    summary = {
        "schema_version": 1,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "minimum_per_persona": args.min_per_persona,
        "renderer_schema": 3 if batch_size > 1 else 2,
        "batch_size": batch_size,
        "personas": manifests,
        "excluded": excluded,
        "all_included_personas_ready": bool(manifests)
        and all(item["ready_for_training"] for item in manifests.values()),
        "warning": "Not a promotion artifact. Train only after each target persona has frozen valid/test splits.",
    }
    write_json(out / "manifest.json", summary)
    print(f"dialogue export: {out} ready={summary['all_included_personas_ready']}")
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

    event_debate = subcommands.add_parser("event-debate", help="run evidence-coded, event-driven local debate")
    event_debate.add_argument("--ledger", required=True)
    event_debate.add_argument("--domain", required=True, choices=sorted(load_domains()))
    event_debate.add_argument("--backend", choices=("mlx", "ollama"), default="mlx")
    event_debate.add_argument("--model-path", help="local MLX model directory")
    event_debate.add_argument("--ollama-model", default=DEFAULT_MODEL)
    event_debate.add_argument("--api-url", default=DEFAULT_API_URL)
    event_debate.add_argument("--timeout", type=int, default=180)
    event_debate.add_argument("--out", default="runs")
    event_debate.add_argument("--max-turns", type=int, default=10)
    event_debate.add_argument("--reconcile-rounds", type=int, default=2)
    event_debate.add_argument("--max-tokens", type=int, default=600)
    event_debate.add_argument("--temperature", type=float, default=0.1)
    event_debate.add_argument("--seed", type=int, default=20260825)
    event_debate.add_argument("--prompt-profile", choices=EVENT_PROMPT_PROFILES, default="orthogonal_fewshot")
    event_debate.add_argument(
        "--adapter-map",
        help="persona idからutterance renderer v3用MLX LoRA directoryへのJSON map（判断には不使用）",
    )
    event_debate.add_argument(
        "--renderer-adapter",
        help="全人格を1バッチ描画する共有utterance renderer v3のMLX LoRA directory",
    )
    event_debate.add_argument(
        "--body-adapter",
        help="検証済みclaim本文だけを生成し、moveをコード合成するclaim-body v2 MLX LoRA directory",
    )
    event_debate.add_argument(
        "--fast",
        action="store_true",
        help="2主張/人格、最大2発言/人格、すり合わせなしでlive shadowを軽量実行",
    )
    event_debate.add_argument(
        "--no-renderer",
        action="store_true",
        help="構造判断とすり合わせを残し、会話rendererだけを検証済みstatement合成へ置換",
    )
    event_debate.set_defaults(handler=run_event_debate)

    rsi = subcommands.add_parser("rsi-shadow", help="gate one bounded prompt/config RSI shadow round")
    rsi.add_argument("--parent-dev", required=True)
    rsi.add_argument("--candidate-dev", required=True)
    rsi.add_argument("--parent-holdout", required=True)
    rsi.add_argument("--candidate-holdout", required=True)
    rsi.add_argument("--round", type=int, default=1)
    rsi.add_argument("--max-rounds", type=int, default=3)
    rsi.add_argument("--out", required=True)
    rsi.set_defaults(handler=run_rsi_shadow)

    mark_event = subcommands.add_parser("mark-event", help="approve or reject one complete event-debate run")
    mark_event.add_argument("run")
    mark_event.add_argument("status", choices=["approved", "rejected"])
    mark_event.add_argument("--reviewer", default="FreeBSE")
    mark_event.add_argument("--note", required=True)
    mark_event.set_defaults(handler=mark_event_run)

    dialogue_export = subcommands.add_parser(
        "export-dialogue", help="export approved event dialogue as per-persona MLX chat JSONL"
    )
    dialogue_export.add_argument("runs", nargs="+")
    dialogue_export.add_argument("--out", default="data/dialogue_sft")
    dialogue_export.add_argument("--min-per-persona", type=int, default=30)
    dialogue_export.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="2以上でruntime同形のutterance renderer v3バッチJSONLを出力",
    )
    dialogue_export.add_argument(
        "--shared-renderer",
        action="store_true",
        help="人格metadata付きの共有renderer v3 datasetも出力",
    )
    dialogue_export.set_defaults(handler=export_dialogue_sft)

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
