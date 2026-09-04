# Horse renderer Weight v1 — 2026-09-03

## Result

General Dialogue v4 shared step160から、競馬固有の限定表現を保持するrenderer-only LoRAを48 iteration継続学習した。一般rehearsalを残したままhorse direct holdoutは0/4から2/4へ、emailは5/16から13/16へ改善し、EVは14/18を維持した。

ただし実際のhorse claim ledger 8 eventでは、厳格validator下のWeight直接採用が0/8だった。安全fallbackは成立したが、**非昇格**である。

- `promotion_allowed=false`
- `parent_replacement_allowed=false`
- `automatic_publish_allowed=false`
- Baseのclaim選択と証拠検証は固定
- Weight、MLX dataset、生成rawはdata4またはhorseの内部learning artifactに保存

## Trigger and validator repair

元のv4 step160は、次の凍結claimを2件とも逆向きに描画した。

```text
claim: 公開3頭BOXの全面置換には使わない
output: 公開3頭BOXを直ちに置換する
```

validatorへ、同一動作語近傍の制限保持、制限省略、対立案の強いdirective混入、助詞で途切れた文末を検査する処理を追加した。正しい混合文と「置換案は危険」のような拒否表現は許可する。29 unit testsは全て成功した。

## Data and training

- human-curated horse: train 16 / valid 4 / test 4
- renderer contract validation: 24/24
- general v4 rehearsal: 153
- combined train: 217（153 + 16×4）
- resumed adapter: `general-dialogue-v4/shared_renderer_step160`
- rank 8 / scale 16 / last 8 layers / LR 1e-5 / 48 iteration
- horse test loss: 4.148 → 1.301
- horse test ppl: 63.318 → 3.674
- adapter: `/Volumes/data4/cod_model_weight/adapters/general-dialogue-v4-horse-v1/shared_renderer_step48`
- SHA256: `fbc5c5da01227884b3792152f9d316a586133d7941d2a4915beedd1bf1b12bb3`

## Functional gates

| frozen check | parent | candidate |
|---|---:|---:|
| horse direct | 0/4 | 2/4 |
| EV direct | 14/18 | 14/18 |
| email direct | 5/16 | 13/16 |
| horse event-debate direct | 0/8 after validator | 0/8 after validator |
| competing claim selection accepted | 0 | 0 |

Lossと小規模holdoutは改善したが、実走の直接描画へtransferしていない。次は自然なhorse `agree / maintain / revise` targetと別topic holdoutを増やし、horse実走で過半数を直接描画できるまで追加Weightを昇格させない。

Horse側の全データ、CoD、専用モデル比較は `/Users/osaka/src/horse/docs/learning/20260903_horse_domain_model_v2/training_resume_report.md` に記録した。
