# Claim Body Device Harness

Qwen3-0.6B-4bit構造Base、Qwen3-1.7B-4bit、Claim Body v3 LoRAを、ExtremeWeather本体へ組み込む前にiOS 17/A15実機で段階検証する独立harnessです。

確認するもの:

- public Hugging Face Baseのdownload / load
- bundled MLX LoRAのload
- 独立`ChatSession`による4人格のtemperature 0、96 tokens以下のclaim-body生成
- strict JSON、ID、claim語彙、D番号非混入
- TTFT、tokens/s、total time、thermal state、task/MLX memory
- session clear後のMLX cache解放
- 実行中cancel、Adapter unload
- 成功metricsをDocumentsの`mp_cod_a15_soak.json`、cancel証跡を`mp_cod_a15_cancel.json`へatomic保存
- Native CoDを`mp_cod_a15_native_cod.json`へatomic保存

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

4人格を独立sessionで連続生成した実測:

| Run | Utterances | Total | Peak footprint | Minimum headroom | Thermal |
|---|---:|---:|---:|---:|---|
| MLX cache解放前 | 4/4 | 12.077秒 | 2,510.879 MiB | 561.137 MiB | fair |
| sessionごとにcache解放 | 4/4（exact 2/4） | 11.904秒 | 1,478.894 MiB | 2,058.169 MiB | nominal |

`Memory.clearCache()`により公開文と速度を変えずpeak footprintを41.1%削減しました。生成開始250ms後の自動cancelも`CANCELLED`となり、Adapter unloadとMLX cache 0を確認しています。

```sh
python3.11 tools/validate_iphone_claim_body_soak.py \
  /path/to/iphone13_a15_four_persona_soak.json \
  --baseline /path/to/iphone13_a15_four_persona_soak_before_cache_clear.json \
  --cancel-result /path/to/iphone13_a15_cancel.json
```

## Native CoD mode

アプリの「A15 Native CoDを実行」、または`--autorun --native-cod`で次を実行します。

1. 0.6B Baseが人格別`role_preferences`内でclaim / D番号 / confidenceを盲検選択
2. 初期多数派の票を保持し、異論側だけBase再選択
3. 見解変更時はBase選択済みの旧claim・新claim・D番号から`change_reason`を決定論合成
4. 全構造判断後にClaim Body v3 LoRAをロード
5. unique claim本文だけを生成し、同一claimはcache
6. objection → revise → agreementの優先順で会話化
7. Adapterをunloadし、memory / thermal / rawを保存

他人格の自然文をBaseへ渡さず、LoRAはclaim・evidence・vote・change reasonを生成しません。構造Baseと本文Baseは同時常駐させず、各session後にMLX cacheを解放します。

```sh
python3.11 tools/validate_iphone_native_cod.py \
  /path/to/iphone13_a15_native_cod.json \
  --repeat /path/to/iphone13_a15_native_cod_repeat.json
```

詳細と全HOLD履歴は[`../../docs/iphone13_a15_native_cod_20260904.md`](../../docs/iphone13_a15_native_cod_20260904.md)を参照してください。

物理iPhone 13 Pro / A15のwarm-cache runは、構造7 model call、公開7 event、17.245秒でhard gateを通過しました。永続cacheを直接再読込した2回目も18.771秒、peak 1,318.144 MiB、headroom 2,068.278 MiB、thermal fairで通過しています。初期3案から異議・見解変更・賛同を経て、最終2対2を`unresolved_tie`として保持しました。

永続本文cacheは、同じ実機で生成したWeight raw、Adapter SHA、claim label、body、canonical digestを保持します。各runで再検証し、完全な時だけLoRAの再load・再生成を省きます。

`--autorun --native-cod --weather-replay`は、bundle内の監査済み台風18号歴史replay JSONを読み込みます。ledger SHA不一致を拒否し、直交する進路・強度・防災claimを単一勝者へ潰さず、明示的な`contradicts`だけを再討論します。物理A15の検証済み3 runは13.631〜13.735秒、fallback 0、thermal nominal、semantic完全一致でhard gateを通過しました。

## Boundary

- 学習しない
- ExtremeWeatherへMLX packageを追加しない
- Base WeightをGit/Appへ同梱しない
- Native CoDは架空の均衡fixtureで検証し、実案件の意思決定結果として使わない
- ExtremeWeather本体への統合は別の回帰・memoryゲートを通す
