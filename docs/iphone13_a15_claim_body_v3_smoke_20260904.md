# iPhone 13 Pro / A15 Claim Body v3 device smoke — 2026-09-04

## Result

物理iPhone上で`mlx-community/Qwen3-1.7B-4bit`をロードし、Claim Body v3 MLX LoRAを適用して1件の本文を生成できた。通常memory limit、temperature 0、最大96 tokensで、strict contract検証、Adapter unload、session clearまで完走した。

```text
input claim: 暴風が強まる前の安全確保を優先する
model output: {"bodies": [{"id": "B01", "body": "暴風が強まる前に安全確保を優先します"}]}
result: PASS
```

## Environment

- device: iPhone 13 Pro / A15
- OS: iOS 17.6.1
- runtime: MLX Swift LM 3.31.4 / MLX Swift 0.31.4
- downloader: swift-huggingface 0.10.0
- tokenizer: swift-transformers 1.3.4
- base: `mlx-community/Qwen3-1.7B-4bit`（端末側cache、約984 MB）
- adapter: Claim Body v3 step128（2,496,303 bytes）
- harness: [`../ios/ClaimBodyDeviceHarness`](../ios/ClaimBodyDeviceHarness)
- bundle: `com.freebse.MPCoDClaimBodyHarness`
- Increased Memory Limit entitlement: なし

## Measurements

| Run | Base load | Adapter load | TTFT | Prompt | Generated | Speed | Total | Thermal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 初回cold download/load | 39.580秒 | 0.053秒 | 4.499秒 | 208 | 30 | 24.415 tok/s | 44.493秒 | nominal |
| warm cache 1 | 2.090秒 | 0.014秒 | 2.325秒 | 208 | 30 | 25.277 tok/s | 4.486秒 | nominal |
| warm cache 2 / saved result | 2.176秒 | 0.021秒 | 2.306秒 | 208 | 30 | 25.732 tok/s | 4.558秒 | nominal |

初回cold値はdevice console、最終warm値はアプリDocumentsへatomic保存したJSONから取得した。

## Evidence

```text
result JSON  /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_smoke_20260904.json
result SHA   c3008004f01ddf941f9b6080e1f8df869ca5c35e8f432a67616a4bebcf9bd8ba
adapter SHA  4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92
config SHA   4e9dc5ffbe79d381b58e931e0a4a4ec5dc5b879786d733fd1212ef7880460e48
```

result JSONは`schema_version=1`、model、Weight SHA、TTFT、token数、tokens/s、total、thermal state、生成rawを保持する。Adapter configはSwift側`LoRAContainer`との互換性のため、対象7 module keyを明示した配布版と同一である。

## Reproduce

```sh
cd /Users/osaka/src/cod_model/ios/ClaimBodyDeviceHarness
cp /Volumes/data4/cod_model_weight/releases/mp-cod-claim-body-v3/adapters.safetensors \
  Resources/Adapter/adapters.safetensors
xcodegen generate
xcodebuild \
  -project ClaimBodyDeviceHarness.xcodeproj \
  -scheme ClaimBodyDeviceHarness \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  -derivedDataPath DerivedData \
  -skipMacroValidation \
  build
```

署名済みappを実機へinstallし、`--autorun`で起動する。初回だけBaseをHugging Faceから取得し、以後は端末cacheを利用する。現行版は成功時に画面とconsoleを`PASS`にし、Documentsへ`mp_cod_a15_soak.json`を保存する。`--cancel-during-generation`を追加するとcancel経路を実行する。

## Boundary

最初のPASSが証明するのは「A15 / iOS 17.6.1で、1.7B 4bit Base + 2.5 MB LoRAによる1件のClaim Body生成が通常memory limit内で動く」ことだけである。

## Four-persona sequential soak

同じWeightを一度だけloadし、各人格に新しい`ChatSession`を割り当てて4件を順次生成した。他人格の発言履歴は次のsessionへ渡していない。

```text
力学モデル研究者: 進路予測は上層場と地上場の整合を確認して更新します
アンサンブル確率予報者: 少数だが重大なシナリオも分布に残して比較します
観測・ナウキャスト専門家: 観測時刻と出典を毎回確認します。
影響・リスク予報者: 暴風が強まる前に安全確保を優先します
```

端末内strict検査に加え、取得後の4文をPython本番`body_matches_claim`、neutral、polite、strict `bodies` schemaへ通し、全件合格した。学習targetと文末記号を除いて完全一致したのは2/4で、残りは意味を保つ自然な省略・言い換えだった。

| Metric | Cache解放前 | Sessionごとに`Memory.clearCache()` | Change |
|---|---:|---:|---:|
| Contract-valid bodies | 4/4 | 4/4 | unchanged |
| Exact polite target | 2/4 | 2/4 | unchanged |
| Total | 12.077秒 | 11.904秒 | -1.4% |
| Decode speed | 約25 tok/s | 25.564 tok/s | unchanged |
| Peak task footprint | 2,510.879 MiB | 1,478.894 MiB | -41.1% |
| Minimum memory-limit headroom | 561.137 MiB | 2,058.169 MiB | +1,497.032 MiB |
| Thermal | fair | nominal | run variance |

未解放時はsessionごとにtask footprintが増えた。`session.clear()`後にMLX buffer cacheを解放すると、各発言後のcurrent footprintは約1.06GBへ戻り、速度と本文は維持された。

```text
baseline JSON  /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_four_persona_soak_before_cache_clear_20260904.json
baseline SHA   3ff050552c9c1b0949582e118193b93a93fdb64e4c54fda6ba8f3104a782884c
optimized JSON /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_four_persona_soak_20260904.json
optimized SHA  8205792a922733004623534431f7f68d8332ae49c4e39aa1dd96d7f018a61fba
```

## Cancellation

`--cancel-during-generation`で最初の生成開始250ms後にtaskをcancelした。MLX streamがcompletion metricsなしで終了する場合も`Task.isCancelled`から`CANCELLED`へ正規化し、session clear、Adapter unload、MLX cache解放後に証跡を保存する。

```text
completed utterances  0
adapter unloaded      true
MLX cache after stop  0 bytes
thermal               nominal
cancel JSON           /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_cancel_20260904.json
cancel SHA            869313c595f312f97c42417ae2b8d196e85485e8cdd05f02eee5098e45d6f1cd
```

検証command:

```sh
python3.11 tools/validate_iphone_claim_body_soak.py \
  /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_four_persona_soak_20260904.json \
  --baseline /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_four_persona_soak_before_cache_clear_20260904.json \
  --cancel-result /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_cancel_20260904.json
```

結果は`status=valid`、4 utterances、exact polite 2、peak 1,478.894 MiB、headroom 2,058.169 MiB、peak reduction 41.101%、cancel validだった。

## Remaining boundary

このsoakは4人格の自然文rendererを連続実行したもので、Baseによるclaim/evidence/vote選択や、反論・賛同・意見変更を伴うreconciliationではない。完全な複数人格CoD、長いcontext、memory warning、3D画面との共存は未検証で、ExtremeWeather本体へMLXを追加する許可にはしない。
