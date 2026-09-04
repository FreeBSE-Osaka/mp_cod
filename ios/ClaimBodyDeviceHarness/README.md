# Claim Body Device Harness

Qwen3-1.7B-4bitとClaim Body v3 LoRAを、ExtremeWeather本体へ組み込む前にiOS 17/A15実機で1回だけ検証する独立harnessです。

確認するもの:

- public Hugging Face Baseのdownload / load
- bundled MLX LoRAのload
- temperature 0、96 tokens以下のclaim-body生成
- strict JSON、ID、claim語彙、D番号非混入
- TTFT、tokens/s、total time、thermal state
- Adapter unloadとsession clear
- 成功metricsをDocumentsの`mp_cod_a15_smoke.json`へatomic保存

## Generate

```sh
cd ios/ClaimBodyDeviceHarness
cp /Volumes/data4/cod_model_weight/releases/mp-cod-claim-body-v3/adapters.safetensors \
  Resources/Adapter/adapters.safetensors
xcodegen generate
```

`adapters.safetensors`、生成`.xcodeproj`、DerivedDataはGit管理外です。

## Build

```sh
xcodebuild \
  -project ClaimBodyDeviceHarness.xcodeproj \
  -scheme ClaimBodyDeviceHarness \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  -derivedDataPath DerivedData \
  -skipMacroValidation \
  build
```

Bundle IDは`com.freebse.MPCoDClaimBodyHarness`、Development Teamは既存ExtremeWeatherと同じ`4RVAYB6R59`です。

通常の開発署名profileにはIncreased Memory Limit entitlementが含まれなかったため、このharnessは通常memory limitで検証します。

`mlx-swift-lm 3.31.4`、`swift-huggingface 0.10.0`、`swift-transformers 1.3.4`へ固定しています。`-skipMacroValidation`は初回CLI buildでSwift macro trust UIを待たないための、そのbuild限定optionです。

## Device result

2026-09-04、iPhone 13 Pro / A15 / iOS 17.6.1で直接推論に成功しました。

| Run | Base load | Adapter load | TTFT | Generate | Total | Thermal |
|---|---:|---:|---:|---:|---:|---|
| 初回cold | 39.580秒 | 0.053秒 | 4.499秒 | 24.415 tok/s | 44.493秒 | nominal |
| cache済みwarm | 2.176秒 | 0.021秒 | 2.306秒 | 25.732 tok/s | 4.558秒 | nominal |

warm runは208 prompt tokensから30 tokensを生成し、`{"bodies":[{"id":"B01","body":"暴風が強まる前に安全確保を優先します"}]}`を返しました。詳細は[`../../docs/iphone13_a15_claim_body_v3_smoke_20260904.md`](../../docs/iphone13_a15_claim_body_v3_smoke_20260904.md)を参照してください。

## Boundary

- 学習しない
- 複数人格討論を実行しない
- ExtremeWeatherへMLX packageを追加しない
- Base WeightをGit/Appへ同梱しない
- 物理端末でPASSしたのは1件の本文生成だけで、完全な複数人格CoDではない
- ExtremeWeather本体への統合は別の回帰・memoryゲートを通す
