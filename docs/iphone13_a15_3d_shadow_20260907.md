# A15 Native CoD × Metal 3D shadow — 2026-09-07

## Result and scope

iPhone 13 Pro / A15 / iOS 17.6.1の独立harnessで、台風18号歴史replayのNative CoDとMetal 3D描画を同じprocess内で実行した。同時実行、3D要求によるcancel、模擬memory warningによるcancelの3試行が外部validatorを通過した。

描画にはローカルExtremeWeatherの`Regional3DMetalVolumeView.swift`から、volume / precipitation shaderをそのまま抽出して使用した。入力は32×32の固定合成雲場と32³の試験noise、描画384×256、24 ray steps、20 fps目標。cameraを連続回転し、volumeとrainの2 passを毎frame実行する。これは本体全体のUnity、ArcGIS地図、通信、実データtextureの負荷を再現したものではない。

BaseはQwen3-0.6B-4bitでclaim/evidenceを実際に生成。Claim Body v3本文6件は、過去の実機Weight rawを再検証した永続cacheを利用した。今回の同時実行中に1.7B/LoRAをloadしたわけではない。全runで`production_integration_allowed=false`を維持する。

## Measurements

| Metric | Concurrent | 3D handoff | Memory warning (simulated) |
|---|---:|---:|---:|
| External gate | PASS | PASS | PASS |
| 3D-only baseline | 20.04 fps | 19.98 fps | 20.05 fps |
| CoD elapsed | 14.834 s | cancelled | cancelled |
| Frames while CoD runs | 285 / 19.20 fps | 0 (paused) | 0 (paused) |
| Inference p95 frame gap | 56.49 ms | n/a | n/a |
| Inference maximum frame gap | 220.73 ms | n/a | n/a |
| Cancellation request → task end | n/a | 1.413 s | 1.384 s |
| Recovery 3D | 20.16 fps | 20.17 fps | 20.11 fps |
| Peak task footprint | 1,165.42 MiB | 963.33 MiB | 962.10 MiB |
| Minimum sampled headroom | 1,906.59 MiB | 2,108.68 MiB | 2,109.90 MiB |
| Post-inference MLX active/cache | 3,168 / 0 bytes | 3,168 / 0 bytes | 3,168 / 0 bytes |

fpsはGPU command完了数/phase実時間。ディスプレイを外部cameraで計測した値ではない。285 concurrent framesは2 passのMetal command完了時刻を保持している。最大221msの間隔があり、無引っ掛かりとは主張しない。

同時実行のCoDは構造5 calls、初期4論点、再討論1名、6 events、本文fallback 0、`unresolved_plurality`。2026-09-05の単独runとledger、初期/最終選択、D番号、confidence、発言、cache snapshotが一致した。測定中thermalはnominal。250ms周期のsampleで、従来のphase境界sampleより小さいheadroomも捕捉している。

## Dialogue provenance

以下は同時実行runの公開`utterance`である。本文はWeight由来cache、C05/C06の接続句はコード合成。現在予測ではなく、2026-08-25 15:50 JST cutoffの資料による歴史的再生である。

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

Codex判断: 描画負荷を加えてもこの固定ledgerの意味出力は保たれた。本体統合の採否は、full 3D workloadとcache miss時の1.7B本文生成を別に測って決める。

## Stops and failures

- 最初の描画callbackはSwift 6でMainActorを継承し、Metal完了queueから呼ばれる際のdata-race warningを出した。途中停止し、`@Sendable` closureからMainActorへmetricsを渡すよう修正。最終3試行にはこのwarningがない。
- 再取得を含むcold試行は86.407秒。そのうちBase loadは72.302秒で、0.6B Weight約320MBを端末へ再取得していた。35秒gateを外部validatorが拒否した。cold JSONは独立保存し、warm性能へ混ぜない。
- cold直後はstart thermalがnominalでなく開始拒否された。冷却後、同じnominal gateで再開。ゲートは緩めていない。
- 開始失敗も`status=hold` JSONへ保存し、端末に残る前回成功JSONとの取り違えを防ぐ。
- メモリ警告は専用metadata付きの`UIApplication.didReceiveMemoryWarningNotification`をアプリ内で送った試験。実際のOS pressure/OOM/jetsam発生を証明しない。
- GPUのdrain待ちは非同期・最大5秒。memory warning、512 MiB未満のheadroom、serious/critical thermal、app非active化でCoD停止を要求する。今回直接実行した停止原因は3D要求と模擬通知の2つ。

## Reproduce and validate

準備、build、launch modeは[ハーネスREADME](../ios/ClaimBodyDeviceHarness/README.md#3d-shadow-on-a-physical-iphone)を参照。

```sh
python3.11 -m unittest -q test_iphone_3d_shadow test_cod_model
python3.11 tools/validate_iphone_3d_shadow.py \
  /path/to/iphone13_a15_3d_concurrent_20260907.json \
  --shader-packet /path/to/iphone13_a15_3d_shader_packet_20260907.json \
  --reference /path/to/iphone13_a15_typhoon18_replay_final_20260905.json
```

同じvalidatorへhandoff / memory warning JSONを渡す。出力はframeごとの時刻から再計算し、MLX解放と3秒以内のcancelを確認する。GPU終了待ちの5秒制限はSwift側で実施する。concurrent modeでは既存`validate_iphone_native_cod.py`も実行し、保存metricsだけでなくraw JSON・D番号・本文意味・cache・単独runとの一致を再検証する。

最終sourceをarm64実機用にbuildし、署名確認、端末install、3試行の実行とresult回収を行った。純Python regressionは34 tests PASS。物理GPUのscene PNGも回収・目視した。

## Audit artifacts

すべて `/Volumes/data4/cod_model_weight/evaluations/claim-body-v3/`。時刻、phase、全frames、memory samples、concurrent modeのnative rawはJSON内にある。

| File | SHA-256 |
|---|---|
| `iphone13_a15_3d_concurrent_20260907.json` | `9cd44bdcaa263558d3a5c53ad68fe02629c35edd5f2d0292506eab0f2ea859ee` |
| `iphone13_a15_3d_handoff_20260907.json` | `6893d227029222d4e7bad05741ca7d025fc231c216b22f3b8d555c309128b098` |
| `iphone13_a15_3d_memory_warning_simulated_20260907.json` | `2712d535e0c0522b58dc6d8f109e9e20fd200c5cd18043518b22e71bd84508c8` |
| `iphone13_a15_3d_concurrent_cold_20260907.json` | `e728c74584e02f4fe8f98d59778e69b184b949e5b5b0958f6cd3b139a7f1afb0` |
| `iphone13_a15_3d_shader_packet_20260907.json` | `80cb2e08285c9ea575c91c4214984b83f07863406ccfbb8ddce7941eb9ea55d8` |

`CoD3DShadow.swift` SHA: `eae99cba8b9a8e5d5482ffd6a835ed60498bc1a5cb1ea9a463146eadd204a9c3`。
ExtremeWeather source SHA: `2f5c22f530b2bb46d21fcd57ff07bcfa709917e9a218f61ff161de48eeddf2b8`。
Extracted shader SHA: `e90a34e5ddd1892de495a6aa5c62d86a945bc8d2a316f5334d71282c4210b595`。

本体の既存追加変更、horseの変更、モデルWeightはこの作業で書き換えていない。ExtremeWeather本体へのMLX追加は未実施。
