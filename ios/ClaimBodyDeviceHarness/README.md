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

## 3D shadow on a physical iPhone

`CoD3DShadow.swift`はExtremeWeatherの現行Metal volume / precipitation shaderを、
固定の合成雲場で動かす独立試験です。384×256、24 ray steps、20 fps目標で、
GPU完了時刻・frame間隔・thermal・task/MLX memoryを保存します。
Unity、ArcGIS地図、通信、実際の気象データを含む本体全体の負荷試験とは区別します。

ローカルのExtremeWeather sourceからshaderを抽出してからprojectを生成します。
元sourceとshaderのSHA-256を保存し、端末でshader hashを再確認します。
準備したshader packetはGit管理外で、他projectのsourceをmp_codへ再配布しません。

```sh
python3.11 tools/prepare_iphone_3d_shadow.py \
  /path/to/typhon_exweather/ExtremeWeather/ExtremeWeather/Features/Regional3D/Regional3DMetalVolumeView.swift
xcodegen generate --spec ios/ClaimBodyDeviceHarness/project.yml
```

上記の実機build/install後、次のlaunch argumentsで実行します。

| Arguments | Test | Documents result |
|---|---|---|
| `--3d-shadow` | 3Dのみ3秒 → CoDと3D同時 → 3Dのみ3秒 | `mp_cod_a15_3d_concurrent.json` |
| `--3d-shadow --3d-handoff` | 3D要求でCoD停止 → MLX解放確認 → 3D再開 | `mp_cod_a15_3d_handoff.json` |
| `--3d-shadow --simulate-memory-warning` | メモリ警告通知を模擬 → 同じ停止処理 → 3D回復 | `mp_cod_a15_3d_memory_warning_simulated.json` |

初回downloadを含む試行とcache済み試行は別ファイルでMacへ回収します。
開始thermalはnominal必須。512 MiB未満のheadroom、serious/critical thermal、
OS memory warning、app非active化は停止理由として記録します。
模擬通知には固有metadataを付け、OSからの実通知と区別します。
開始できなかった場合も`status=hold`のJSONで前回結果を置き換えます。
各再実行前に必要な端末resultを回収してください。

```sh
python3.11 tools/validate_iphone_3d_shadow.py /path/to/device_result.json \
  --shader-packet ios/ClaimBodyDeviceHarness/Resources/Shadow/shadow_renderer.json \
  --reference /path/to/accepted_native_typhoon18_replay.json
```

外部validatorは描画15 fps以上、最大stall 1秒以下、同時実行時のp95 gap 150ms以下、
CoDの既存contract/35秒gate、512 MiB headroom、停止後MLX cache 0を検査します。
切替では3Dが推論中に動かないこと、3秒以内のcancel、解放後の描画回復を確認します。
rawの`production_integration_allowed=false`は、この試験に通っても維持します。

### Body-cache miss

各modeに`--fresh-body`を追加すると、既存の本文cacheを読み書きせず、0.6Bの判断後に
1.7B + Claim Body v3をloadし、全6 claimをその場で生成します。同一run内の再利用は維持します。
result filenameには`_fresh_body`を付け、通常cache/resultを保護します。

`--3d-shadow --fresh-body`は3Dと本文生成を同時実行します。
`--3d-shadow --fresh-body --3d-handoff`と
`--3d-shadow --fresh-body --simulate-memory-warning`は、最初のLoRA本文生成開始後に停止を要求します。
`stop_inference_progress`で構造判断より後の本文生成まで到達したことを検査します。

同じvalidatorが`fresh_body=true`を識別し、body model/LoRA load、全claimの実生成raw、
`body_cache_bypassed=true`、`body_cache_persisted=false`を必須にします。
処理時間は追加の本文生成を含め60秒、cache利用時は従来の35秒です。
描画15 fps、512 MiB headroom、cancel 3秒、解放後のcache 0は共通です。
`--reference`ではBaseの判断一致を必須にし、新しく生成した本文の字面の一致は別指標として記録します。
データを含む最初のmodel downloadはwarm性能と別runに保存してください。

## Boundary

- 学習しない
- ExtremeWeatherへMLX packageを追加しない
- Base WeightをGit/Appへ同梱しない
- Native CoDは架空の均衡fixtureで検証し、実案件の意思決定結果として使わない
- ExtremeWeather本体への統合は別の回帰・memoryゲートを通す
