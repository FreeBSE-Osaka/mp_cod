# General Dialogue utterance renderer v3 — 2026-08-29

## Result

LoRAをclaim選択、証拠選択、confidence、statement、すり合わせ投票から外し、検証後の構造化recordを`utterance`へ変換するrenderer専用構成を実装した。Weight本体は複数作成できたが、未学習payloadでmoveと選択を安定して保持できなかったため、**全候補を非昇格**とした。

- `promotion_allowed=false`
- `parent_replacement_allowed=false`
- `automatic_publish_allowed=false`
- Base本体は全実験で固定
- 現行運用はBase構造判断 + validator。`--fast`では未昇格rendererを呼ばない

## Runtime separation

構造判断が返すキーは次に限定した。

- 独立主張: `code`, `data_ids`, `confidence`, `statement`
- すり合わせ: `choice`, `data_ids`, `statement`, `change_reason`
- renderer: `utterances`配列内の`id`, `utterance`だけ

MLX Adapterはrenderer callの直前だけロードし、構造判断前には必ずLoRA層を外す。rendererは入力IDの欠落、重複、未知ID、余分なJSONキー、D番号、内部protocol、未完全文、move不一致、選択済みでない競合案の採用を拒否する。1 batchは学習分布と同じ最大3発言に制限した。

## Frozen data

承認済み8runの131発言を使用した。fallback、不正文、機械文、近似同文は除外済みである。

| Persona | Train utterances | Valid | Test | Total |
|---|---:|---:|---:|---:|
| empirical_auditor | 27 | 3 | 3 | 33 |
| falsifier | 27 | 3 | 3 | 33 |
| hypothesis_builder | 27 | 3 | 3 | 33 |
| pragmatic_operator | 26 | 3 | 3 | 32 |

共有rendererは各人格のsplitを混ぜずに結合し、train 107 / valid 12 / test 12発言、36 / 4 / 4 batchとした。最終の軽量共有schemaは長い効用・損失文をitemから外し、`speaker`, `move`, `speech_act`, 凍結済み主張・証拠だけを渡す。

```text
dataset manifest SHA-256 5d6f3c7f9b42951c77ebd037e6ed193f10fe57b0fae42b904e129200f00cbcf9
```

## Training

設定は [`configs/general-dialogue-v3.yaml`](../configs/general-dialogue-v3.yaml)。Qwen3-1.7B MLX 4bitをBaseに、LoRA rank 8 / scale 16 / last 8 layers、micro batch 1、gradient accumulation 4、learning rate `5e-5`で学習した。trainable parameterは2.490M / 1,720.575M（0.145%）、peak memoryは3.70〜3.99GBだった。

| Candidate | Iter | Final valid | Test loss | Test ppl | Functional result |
|---|---:|---:|---:|---:|---|
| pragmatic_operator | 96 | 0.063 | 0.017 | 1.017 | frozen 3/3、未学習EV direct 1/2。object不合格 |
| shared full | 96 | 0.366 | 0.074 | 1.076 | frozen direct 9/12、未学習5件で反復崩壊 |
| shared full early | 24 | 0.877 | 0.581 | 1.789 | schema 17/17、direct 3/17 |
| shared lite early | 24 | 0.854 | 未計測 | 未計測 | schema 17/17、direct 4/17、pipeline 9/17 |
| shared lite | 48 | 0.577 | 0.280 | 1.324 | schema 17/17、direct 1/17、pipeline 2/17 |

Baseは共有schema評価17発言の全てでschema不合格だった。LoRAはJSON契約を学んだためv2の入力復唱からは改善したが、話者・move・選択の機能transferが昇格水準に達していない。lossやschemaだけで昇格させない。

## Fast path benchmark

未学習EV fleet台帳をQwen3-1.7B Baseで実行した。

```text
mode: --fast, no renderer Adapter
independent model calls: 4
model claims: 8/8
rejected claims: 0
renderer calls: 0
reconciliation calls: 0
elapsed: 29.5 seconds
```

`--fast`は各人格2主張、最大2発言/人格、すり合わせ0回とし、検証済みstatementからD番号を除いた表示文を使う。独立性、claim allowlist、D番号validatorは維持するが、自然文Weightと合意形成を省くためhard gateは通さない。live shadow専用である。

同じBaseで通常版を2 event・1 reconciliation roundまで通した実走は11 call、68.8秒だった。構造投票4件とchange reasonは全てBaseで生成され、rendererはその後に3件ずつ呼ばれた。未昇格rendererを指定しなかったため表示文は安全なstatement fallbackとなった。

## Weight hashes

全Weightは約9.6MBで、リポジトリには含めない。

```text
config                                        dcaf890fc18a7aad1a916365e3a36c92edf4f4e42bdd5c1b391bb9df19b0511b
pragmatic_operator config                     1ac871c1bf62a284f05e3262ab29b767b116e1f757ebb177fa0d10ae1f20d152
pragmatic_operator weights                    89084078a040d390e70f0c0e7203569c6388d29386356f00f2d069cdfc359712
shared full config                            6ae11fce820344045b3e3f0753864be702ed06d3dfb1dce4c365e58027ced7d3
shared full weights                           f5e51d5e4a919a7348d63ad391d7a4f25ad48cceaad74354ae7f5fba3ea7ccb4
shared full step24 config                     aae4b434286a9ef314bb1e2c2521b1d8b6a740006c13127b6b8dbb771e50358b
shared full step24 weights                    51ecf23ce57ff60632197379e6bda2437bb15123d9d020ed215b57d5d4ccbe46
shared lite step24 config                     1afb5a7d6189edfaeb7c6dcf5327d100907524365a6c1d5fffa1a1649f930abf
shared lite step24 weights                    4b6d785bc688dd525bc6305179ab8c4cfd80802505a568d798744bbf31f0062f
shared lite step48 config                     1afb5a7d6189edfaeb7c6dcf5327d100907524365a6c1d5fffa1a1649f930abf
shared lite step48 weights                    383488d3d85c448b2a80a4d84b104928cb91a121b9021c096064aaaca32d6229
```

## Next gate

追加iterationは行わない。再開条件は、各persona×moveに人間確認済みの未重複targetが揃い、別topicの凍結holdoutでBaseより直接valid率が改善し、競合案選択0件・既存能力非劣化を同時に満たせることとする。
