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

署名済みappを実機へinstallし、`--autorun`で起動する。初回だけBaseをHugging Faceから取得し、以後は端末cacheを利用する。成功時は画面とconsoleを`PASS`にし、Documentsへ`mp_cod_a15_smoke.json`を保存する。

## Boundary

このPASSが証明するのは「A15 / iOS 17.6.1で、1.7B 4bit Base + 2.5 MB LoRAによる1件のClaim Body生成が通常memory limit内で動く」ことだけである。完全な複数人格CoD、長いcontext、連続round、peak memory、memory warning、3D画面との共存は未検証で、ExtremeWeather本体へMLXを追加する許可にはしない。
