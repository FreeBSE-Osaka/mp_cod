# A15 / Metal 3D / fresh Claim Body v3 — 2026-09-08

## Result

物理iPhone 13 Pro / A15で、既存の本文cacheを読まず、0.6B Baseの構造判断後に1.7B + Claim Body v3から全6本文を生成した。3D同時実行、本文生成中の3D切替cancel、本文生成中の模擬memory warning cancelの3試行が外部validatorを通過した。

実機情報はiOS 17.6.1 / build 21G101。MLX Swift 0.31.4、MLX Swift LM 3.31.4、swift-huggingface 0.10.0、swift-transformers 1.3.4を使用した。

前回と同じExtremeWeather Metal volume / rain shader、384×256、24 ray steps、固定合成雲場、20 fps目標の独立harnessである。Unity・ArcGIS地図・通信を含む本体全体の負荷試験とは区別し、`production_integration_allowed=false`を維持する。

## Measurements

| Metric | Concurrent fresh body | Handoff during LoRA | Simulated warning during LoRA |
|---|---:|---:|---:|
| External gate | PASS | PASS | PASS |
| Native CoD total | 36.041 s | cancelled | cancelled |
| Structure calls | 5 | before body load | before body load |
| Fresh body calls | 6/6 valid | first generation cancelled | first generation cancelled |
| Public events | 6 | no final publication | no final publication |
| 3D during inference | 706 frames / 19.56 fps | paused | paused |
| Inference p95 / max frame gap | 58.26 / 221.45 ms | n/a | n/a |
| Request → cancellation complete | n/a | 2.241 s | 0.935 s |
| Recovery 3D | 20.08 fps | 19.91 fps | 19.89 fps |
| Peak task footprint | 1,549.07 MiB | 1,471.11 MiB | 1,487.16 MiB |
| Minimum sampled headroom | 1,523.08 MiB | 1,601.00 MiB | 1,612.25 MiB |

モデル取得済みのconcurrent runは、structure load 1.832秒、body model load 2.068秒、LoRA load 0.018秒。本文6件中3件はplain formを安全に丁寧語化した。fallback 0、初期4論点、選択的再討論1名、`unresolved_plurality`。開始thermalはnominal、回復phase終了時はfairで、serious/criticalはなかった。

前回の本文cache利用run（14.834秒、19.20 fps）に対し、本文を6件生成するため実行時間が増えた。Baseのclaim、D番号、confidence、最終票、未解決結論は単独runと一致し、公開6発言の文字列も一致した。今回生成した全6本文のrawが過去cacheと同一であるという主張ではない。

handoff試行のinference phase全体は57.947秒で、停止要求前の構造判断とモデル準備を含む。2.241秒は本文生成中に停止を要求してからTaskが終了するまでの時間である。

fpsはGPU command完了数から計算する。221msの最大frame間隔があり、常時滑らかな画面を保証しない。1回の完走と2種類の停止試験であり、長時間連続実行は未検証。

## Cache and memory lifecycle

`--fresh-body`は本文cacheの読み書きを回避し、同一run内だけで生成済み本文を再利用する。6件を生成してから6 eventsに利用するため、`body_renderer_model_calls=6`かつ`body_renderer_cache_hits=6`となる。後者は保存済み本文の再利用ではない。

- `body_cache_bypassed=true`
- `body_cache_persisted=false`
- `body_model_loaded=true`
- `body_adapter_loaded=true`
- `body_cache_prime_mode=true`

既存`mp_cod_typhoon18_body_cache.json`は試験前後に端末から回収し、`cmp`でバイト一致を確認。共通SHAは`ca2c17ce2128f15b7b73c06dcfe6b5f4ad5c78697826b304d544f472d5825820`。通常resultも上書きせず、fresh resultには`_fresh_body`を付ける。

0.6Bの参照を解放した後のMLX activeは3,168 bytesで、その後1.7Bをloadする。LoRAを外した直後はBaseがまだscope内にあり、MLX active約970MBが残る。呼び出しTaskの終了と参照解放まで待つと、MLX active 6,328 bytes、cache 0となる。concurrent runのprocess footprintも、終了直後約799MiBから3D回復phase終了時約56.91MiBへ下がった。

本体接続時は、Adapter unloadだけを「推論資源の解放完了」と扱わず、Task終了・Base参照解放・残量確認の後で3Dへ戻す必要がある。今回の固定条件では最低512MiB headroomのgateを通過したが、長いpromptや本体の追加textureを含む容量保証にはしない。

## Model text and dialogue

以下は今回のWeight出力の原文。promptのspeakerはcache prime担当の力学モデル研究者で、本文の生成対象は凍結済みの各claimである。4人格の独立した判断は、その前の0.6B呼び出しで行っている。

```json
{"bodies": [{"id": "B01", "body": "北東転向外れを意思決定上の独立シナリオとして残す"}]}
```

公開表示では語尾を「残します」にし、反論・維持の接続句をコードで付ける。以下はその公開`utterance`全文。2026-08-25 15:50 JST cutoffの台風18号資料による歴史的再生であり、現在予測ではない。

```text
力学モデル研究者:
72時間の主判断では北東転向外れの優先度を低くします。

アンサンブル確率予報者:
北東転向外れを意思決定上の独立シナリオとして残します。

観測・ナウキャスト専門家:
上陸前は乾燥空気と鉛直シアで緩やかに弱化します。

影響・リスク予報者:
25日夜から26日朝の奄美と沖縄の暴風高波を最優先します。

アンサンブル確率予報者:
その結論には異議があります。北東転向外れを意思決定上の独立シナリオとして残します。

力学モデル研究者:
結論は変わりません。72時間の主判断では北東転向外れの優先度を低くします。
```

Codex判断: 保存済み本文がなくても、同じ構造判断を保ち、Weightによる本文生成と固定負荷3Dを同時実行できた。本体では本文cacheを優先し、cache miss時は実測したload/生成コストと3D資源の排他・残量条件を考慮する。

## Reproduce

[ハーネスREADME](../ios/ClaimBodyDeviceHarness/README.md#body-cache-miss)の手順で実機build/installする。

```text
--3d-shadow --fresh-body
--3d-shadow --fresh-body --3d-handoff
--3d-shadow --fresh-body --simulate-memory-warning
```

```sh
python3.11 -m unittest -q test_iphone_3d_shadow test_cod_model
python3.11 tools/validate_iphone_3d_shadow.py \
  /path/to/iphone13_a15_3d_concurrent_fresh_body_20260908.json \
  --shader-packet /path/to/shadow_renderer.json \
  --reference /path/to/iphone13_a15_typhoon18_replay_final_20260905.json
```

35 Python tests PASS、arm64実機build/sign/install PASS、3試行のraw回収と外部validator PASS。最終logにdata-race warningやHOLDはない。実機GPUで描画したPNGも回収・目視した。

cache missの60秒gateは追加の本文生成を想定して測定前に設定し、cache利用時の35秒gateは維持。描画15 fps、headroom512 MiB、cancel3秒、解放後cache 0も維持した。比較元もhard gate済みrunであることを必須にしている。

警告試験はmetadata付きのアプリ内通知による模擬であり、OSからの実pressure/OOMを発生させた試験ではない。停止位置は`stop_inference_progress=本文cache prime: TRACK_MAIN_WSW`で、LoRA load後の生成に到達したことを確認した。

## Artifacts

保存先: `/Volumes/data4/cod_model_weight/evaluations/claim-body-v3/`

| JSON | SHA-256 |
|---|---|
| `iphone13_a15_3d_concurrent_fresh_body_20260908.json` | `3b65cbdea65367e28a928f1ce428101c0b9a8e5f08d98c77a6c869f0ed68ed1f` |
| `iphone13_a15_3d_handoff_fresh_body_20260908.json` | `e4f50c6e418d8cd849423beb797afbddc188b63f12cd5595050c48f8f14a8b7d` |
| `iphone13_a15_3d_memory_warning_fresh_body_20260908.json` | `fcb852358075ea11f9abf39484f556bfa2ef06512234afdb5f03cbf1f1112b31` |

Source SHA:

- `NativeCoDSmoke.swift`: `cf5d6a38a91375b34e8e3717d915983a2193be2ca690b8f92066f9421df23a44`
- `CoD3DShadow.swift`: `687f53c116f10db32539efc04427d10d6338ff6eb3e673f67a5257a0f691b19b`
- Shader: `e90a34e5ddd1892de495a6aa5c62d86a945bc8d2a316f5334d71282c4210b595`

WeightはClaim Body v3 step128の同じSHAを使用した。新しい学習やHFへのWeight uploadは行っていない。
