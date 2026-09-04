# Composed statement renderer v1 — 2026-09-04

## Result

発話行為をLoRAへ任せず、コードでmove導入句を確定し、Baseが生成してD番号検証済みの`statement`本文を接続する経路を実装した。`--no-renderer`では独立主張とすり合わせ投票を維持したまま、会話renderer callだけを0にできる。

- claim、evidence、confidence、vote、change reasonは従来どおりBaseが生成
- object / agree / maintain / reviseは複数の導入句を順番にローテーション
- 本文は検証済みstatementからD番号句だけを除去
- 合成後もmove・claim・競合案・制限反転・数値groundingを検証
- 合成不能時だけ従来のlabel templateへ戻る
- `composed_statement_fallback`をWeight由来と数えず、hard gateは通さない

## CLI

```sh
<mlx-python> cod_model.py event-debate \
  --ledger <claim-ledger.json> \
  --domain general \
  --model-path <base-model> \
  --max-turns 12 \
  --reconcile-rounds 2 \
  --no-renderer
```

`--no-renderer`は`--adapter-map`、`--renderer-adapter`と併用できない。`--fast`と違い、主張数、イベント数、すり合わせラウンドを縮小しない。

## EV runtime benchmark

同じQwen3-1.7B MLX 4bit、未学習EV ledger、2 event、1 reconciliation roundで比較した。

| Mode | Model calls | Elapsed | Reconciliation |
|---|---:|---:|---|
| 従来Base renderer | 11 | 68.8秒 | 4人格・1 round |
| `--no-renderer` | 8 | 39.2秒 | 4人格・1 round |

`--no-renderer`はevent renderer 1 callとreconciliation renderer 2 callを削除した。4人格の構造投票、1件の見解変更理由、3/4合意は維持された。

## Live dialogue

```text
批判的設計者:
利用不能が集中する寒冷depotでsmart charging pilotを先に行うと判断します。

仮説構築者:
その結論には異議があります。代案として、電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開すると判断します。

実証監査者:
その案には賛成です。加えて、『電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開する』を採ります。

批判的設計者:
先ほどとは結論を変えます。『電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開する』を採ります。
```

元statementが`label + を採ります`を返した場合は、labelを括弧で囲んで`『…する』を採ります`とし、`…するを採ります`を防ぐ。

## Fast path

`--fast --max-turns 4`は従来どおりすり合わせを省き、4 model call / 22.3秒だった。異議文は同じcomposed statementを使用する。

## Boundary

この経路はWeightモデルではない。後続の[Claim Body Weight v3](claim_body_weight_v3_20260904.md)は未学習3 topicでtransferし、自然文が必要な時の条件付きWeightになった。低遅延を優先する場合は引き続きBase構造判断 + `--no-renderer`、最小遅延が必要な時だけ`--fast`とする。
