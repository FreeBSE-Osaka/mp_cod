# Claim Body Weight v1 — 2026-09-04

## Result

発話行為まで学習させたv3〜v6を止め、LoRAの仕事を「検証済みclaimを自然な一文へする本文」だけに縮めた。Qwen3-1.7B-4bitへ64 iteration学習した実Weightは、学習に使っていない3 topic群の厳格claim一致でBaseの`26/46`から`40/46`へ改善した。

このWeightは次の条件付きで利用できる。

- claim、evidence、confidence、vote、change reasonはAdapterを外したBaseが決める
- object / agree / maintain / reviseの発話行為はコードが付ける
- 本文はclaim一致、制限反転、未入力数字、競合claim、会話move混入を検査する
- 不合格は検証済みstatementへ必ずfallbackする
- safe schema補正を使った発言はWeight由来として記録するが、hard gate通過には数えない
- Base置換、自動公開、構造判断へのAdapter適用は許可しない

## Training

6つの承認済みGeneral v2 topicをtrainに使い、email triageをtopic単位でholdoutした。各exampleは1 itemで、入力は`speaker / claim / evidence`、出力は`id / body`だけである。moveと賛否は含めない。

| Split | 件数 |
|---|---:|
| Train | 96 |
| Valid | 8 |
| Test | 8 |

実行時は`configs/general-dialogue-v2.yaml`へCLI overrideを渡したため、生成済み`adapter_config.json`には元config名が残る。[`configs/claim-body-v1.yaml`](../configs/claim-body-v1.yaml)はその実効値を再現用に固定したもの。Baseからlast 4 layers、rank 4、batch 1、gradient accumulation 4、learning rate `2e-5`で64 iteration学習した。

| 指標 | 値 |
|---|---:|
| 初期valid loss | 2.051 |
| step 32 valid loss | 1.362 |
| step 64 valid loss | 0.396 |
| 最終train loss | 0.317 |
| test loss / perplexity | 0.307 / 1.360 |
| peak memory | 1.822 GB |
| Adapter Weight | 2,496,303 bytes |

## Unseen transfer

保存済みrawを、runtimeと同じsingle-ID schema補正、句点正規化、`body_matches_claim`、制限保持、move非混入で再判定した。どのtopicもtrain 6 topicには含まれない。

| Holdout | Base | Body Weight | 差 |
|---|---:|---:|---:|
| EV smart charging | 12/18 | 14/18 | +2 |
| Email triage | 5/8 | 7/8 | +2 |
| Bike rebalancing | 9/20 | 19/20 | +10 |
| 合計 | 26/46 | 40/46 | +14 |

schemaだけを見るとEVはBase `16/18`、Weight `18/18`が安全なsingle-ID形式へ正規化可能だった。Email Weightは`8/8`が要求どおりの`bodies` schema、Bike Weightは`19/20`が直接またはsingle-ID補正で取得できた。schema適合だけでは昇格せず、上表はclaim自体を保った件数である。

## Runtime smoke

未学習EV台帳、2 event、すり合わせ1 roundを実行した。構造生成8 callと本文生成6 callの合計14 call、46.3秒だった。公開6発言中5件をWeightから採用し、1件は根拠文だけを返してclaimを落としたためvalidatorが拒否し、検証済みstatementへ戻した。

初回実走では長いreconciliation IDの末尾をWeightが省略し、round本文が`0/4`になった。モデルへだけ短い`B01`形式を渡し、元のrecord IDを監査ログへ別保存するよう修正したところ、同条件でround本文は`4/4`採用になった。

| 同じEV 2 event / 1 round | Model calls | Elapsed |
|---|---:|---:|
| 従来Base全文renderer | 11 | 68.8秒 |
| `--no-renderer` | 8 | 39.2秒 |
| Claim Body v1 | 14 | 46.3秒 |

本文は1発言1 callだが約1.1〜1.2秒で、全文rendererより22.5秒短い。`--no-renderer`より7.1秒遅いため、live競馬のように遅延最優先なら引き続き`--fast`または`--no-renderer`を使う。

```text
批判的設計者:
利用不能が集中する寒冷depotでsmart charging pilotを先に行うと判断します。
  origin=composed_statement_fallback
  warning=body renderer does not match the frozen claim

仮説構築者:
その結論には異議があります。代案として、電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開する。
  origin=model_body_v1_schema_repair

実行設計者:
fleet運用担当を置いて寒冷depot限定pilotを実施する。
  origin=model_body_v1_schema_repair

実証監査者:
私も賛成です。特に、電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開する。
  origin=model_body_v1_schema_repair

実行設計者:
その案には賛成です。加えて、電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開する。
  origin=model_body_v1_schema_repair

仮説構築者:
結論は変わりません。電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開する。
  origin=model_body_v1_schema_repair

批判的設計者:
考え直しました。電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開する。
  origin=model_body_v1_schema_repair
```

異議・賛同・維持・変更はWeightに生成させず、検証済みmoveからコード合成した。したがって批判的設計者が見解を変えても、Weightが投票を反転させたわけではない。このrunでは4人格がFULL_SMART_CHARGING_ROLLOUTを選び、3/4条件を満たして対立を解決した。

## CLI

```sh
<mlx-python> cod_model.py event-debate \
  --ledger <claim-ledger.json> \
  --domain general \
  --backend mlx \
  --model-path <Qwen3-1.7B-4bit> \
  --body-adapter <claim-body-v1-adapter> \
  --max-turns 12 \
  --reconcile-rounds 2
```

`--body-adapter`は`--adapter-map`、`--renderer-adapter`、`--no-renderer`と併用できない。Ollama/GGUFへMLX Adapterを直接渡すこともできない。

## Artifact hashes

```text
dataset manifest  b3e2e249ae6096a240df68060e8456bae456c98c2308cb22eb4684c7bdec1a51
training config   7d4fe259ba84fd7eb0236921dbfe9d55846d9100e7bea6c8a1601514d9fef777
adapter config    e2fd8b28d01ab5d598302fc287947653008eaaa8dfb689a46f27aa27dbbedaa8
adapter weights   4e2b90a8e80641eec2874179f72b0fe8bacc0b34d0d585a9afd1d35c67bb08c4
runtime smoke     8facd7cafaf7fdc052861a33796b7ef538a72dea30aa933256ac515c33246c08
```

Weight、dataset、評価rawはdata4に置き、Gitには含めない。公開時はAdapterを別Hugging Face artifactにし、このGitHubリポジトリにはコード、設定、評価境界だけを置く。

ローカルの最終実走rawは`/Volumes/data4/cod_model_weight/evaluations/claim-body-v1/runtime_round1.json`に保存した。
