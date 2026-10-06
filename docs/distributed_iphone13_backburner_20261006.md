# iPhone 13 ProとMacの分散推論実験

2026-10-06、Macと物理iPhone 13 Proの両方で、独立した本文生成jobの分担と、BackburnerによるQwen3.5の層分担を実行した。両方式とも実機動作は確認できたが、独立job分担は今回の小さな仕事では遅くなった。層分担はbatch調整後に時間短縮の兆候があり、追加検証する研究候補として残す。通常のCoD実行経路、採用済みWeight、学習設定は変更しない。

これは推論の分担であり、MLX LoRA学習の分散化ではない。人格別の独立推論を複数の端末へ割り当てる方式と、1回の推論の途中の層を別端末へ渡す方式も区別する。

## 実機と運用条件

MacはApple M2、24 GiB RAM、macOS 15.5、Xcode 16.4。phoneはiPhone 13 Pro、A15、iOS 17.6.1で、既に有効なDeveloper Modeと開発署名を使った。Simulatorの時間は実機測定へ混ぜない。

iPhone上の協力アプリが仕事を受け取り、自分のMetal GPUで計算する。MacがiOSの保護を破ってGPUを直接操作する構成ではない。今回のiOS 17アプリは画面をロックせずforegroundで実行する。OS更新、jailbreak、Wi-Fi pairing、OS全体のwired memory設定変更、他アプリの強制終了は行わない。

モデルとraw記録は外付け`/Volumes/data4`へ保存した。容量監視はMacの内蔵と外付けの双方に20 GiB以上の余裕を要求する。iOS frameworkのビルドはNTFS上の作業ファイル書き込みで停滞したため、ビルド作業場所だけAPFSへ移して完走した。停滞時のログは保存している。

## 独立した仕事の分担

既存の[Claim Body Device Harness](../ios/ClaimBodyDeviceHarness/README.md)へ固定jobの部分実行modeを追加した。Macは4job全てのbaselineと、job0・2だけの分担処理を実行し、phoneはjob1・3を独立sessionで生成する。Mac側の調整役は[distributed_phone_trial.py](../tools/distributed_phone_trial.py)。BaseとAdapterを同時に複数のMacプロセスへロードしない。

両側はQwen3-1.7B-4bitとClaim Body v3 Adapterを使う。同じsystem、固定claim、temperature 0、最大96 tokensで比較する。Base Weight SHA256は`0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f`、Adapter SHA256は`4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92`で一致した。Python MLXとSwift MLXの異なるruntime間のbitwise一致は要求しない。

phoneはUSBで配置した`Documents/DistributedBodyModel`だけを読み、自動downloadしない。requestごとのUUID、job番号、Weight SHA、生成raw、Adapter unload、cache解放、thermal、memory余裕を検査する。UUID結果の再利用と上書きを拒否し、app非active化やmemory warningで停止する。任意のshell命令や文章を受け付ける汎用遠隔APIではない。

1 roundの実測は次の通り。初回のBase転送は含まないが、phoneのアプリ起動と結果回収は分担時間に含む。

| 処理 | wall time | 本文契約 |
| --- | --- | --- |
| Mac単独 全4job | 4.458秒 | 2/4合格 |
| 分担のMac側 2job | 2.437秒 | 2/2合格 |
| 分担全体 Macとphone | 11.821秒 | phoneは2/2合格 |

phoneの内部処理は10.123秒、peak footprint約1.45 GiB、最低headroom約2.01 GiB、thermal nominalだった。phoneが実際に生成した本文は以下で、テンプレートによる代替文ではない。

> 少数だが重大なシナリオも分布に残して比較します
>
> 暴風が強まる前に安全確保を優先します

Mac単独では1件が丁寧語の条件、別の1件が「優先」の保持条件に落ちた。分担は速度も品質同等の比較条件も満たさず、この結果を効率改善や全CoDの完成とは数えない。Macが他の仕事で飽和している場合の容量分担や、常駐workerによる起動コスト削減は別の試験になる。

再実行には、同じBaseとAdapterをphoneとMacへ用意し、phoneのローカル配置を確認してから次を使う。出力先は存在しない新しいディレクトリ、親ディレクトリは既存とする。

```sh
python3.11 tools/distributed_phone_trial.py \
  --device '<CoreDevice ID>' \
  --model /path/to/Qwen3-1.7B-4bit \
  --adapter /path/to/claim-body-v3-adapter \
  --rounds 2 --out /path/to/audits/new-independent-trial
```

## Backburnerの末尾層分担

[Backburnerの公式対象](https://github.com/StayLameBro/backburner/blob/main/docs/INSTALL-IPHONE.md)はiPhone 15 Pro以降などで、[公開ベンチマーク](https://github.com/StayLameBro/backburner)はM4 ProとA18 Pro・A19 Proなどの環境である。今回のA15とLightningは対象外なので、公式の27Bモデルや10 Gb/s USB-Cの速度値を流用しない。

[A15向け互換patch](../patches/backburner-a15/README.md)を、固定したBackburner本体とllama.cpp submoduleへ適用した。新しいSDK enum名を旧SDKでコンパイルできる表現へ変え、runtimeのavailability条件は維持する。使えないSME2経路は明示的に無効化し、誤って呼ばれた場合は計算値を偽造せず失敗させる。通常のiOS memory entitlementで署名した。

ローカルにあったQwen3.5-4B GGUFから、不要なvision・MTPを省いたdecoder-onlyの新規コピーを作った。Mac側は2,647,619,360 bytes、SHA256は`cf7ff1f4df87c2e900a6cde6c52e274977b0eb4cba587434ddb16093a01fadc4`。phone側には同じ元モデルの層28〜31、54 tensor、約275 MiBのheadなしtailを配置した。旧GGUFのRoPE3要素、SSM tensor名、GDN層のKV-head metadataに対応するloader修正を両側で共用した。元Weightは変更せず、他エンジンとの数値同等性や一般的な精度は未検証である。

Macは層0〜27、phoneは転送されたprompt chunkの層28〜31をGPUで処理する。最後のchunkと出力head、通常decodeはMac側に残る。長いKVのphone保持、ANE、split decode、DFlash、LoRAは今回使わない。元のraw portのケーブル制限を維持し、Mac APIは`127.0.0.1`だけへbindする。Wi-Fi gateを解除しない。

### 同一入力の測定

[backburner_phone_trial.py](../tools/backburner_phone_trial.py)は同じpatched binary・同じdecoder GGUF・同じ485-tokenの合成資料入力を、context1024、最大32生成tokens、temperature 0、seed固定、prompt cache無効で測定する。各server起動後の1回はwarmupとし、2回の測定を残す。さらにMac→phoneとphone→Macの順序で試し、設定ごとに各4測定を比較した。転送と状態mergeはrequest時間に含むが、アプリ・serverの起動、Weight転送、hash計算、warmupは表の時間に含まない。

| batchとubatch | Mac単独の中央値 | phone分担の中央値 | 判定 |
| --- | --- | --- | --- |
| 256と64 | 7.324秒 | 8.713秒 | 時間短縮を再現せず |
| 512と128 | 5.072秒 | 4.349秒 | 約14.3％短い研究候補 |

通常条件のMac単独は5.481〜8.874秒、分担は8.195〜8.922秒。調整条件のMac単独は4.265〜6.656秒、分担は3.110〜5.294秒だった。調整条件の単純なspeedup比は1.166倍だが、標本が小さく、無制御の他プロセス負荷と実行時変動がある。安定した全CoD高速化、電力効率改善、常駐運用の効果としては認定しない。batch変更によるMac単独の改善と、phoneを追加する効果も分ける。

phone側の計12 request、warmupを含め42 chunk・3456 remote tokensの計算と状態mergeを確認した。各requestでphoneのchunk増加とMacの`split_finish`完了を対応させ、fallback・Mac側再実行は0。全24 requestの返却生成token列10個は一致した。phoneの24境界サンプルはthermal nominal、footprint最大151.314 MiB、headroom最低2920.686 MiBだった。このfootprintは境界サンプルの最大値で、連続測定した真のGPU memory peakではない。

返却原文は全て次だった。

> 測定していない効果は確認済みとは言えません。

質問の否定内容は保持したが、指定した「いいえ、」の書き出しは守らなかった。このprobeは通信・計算・greedy token一致の確認で、モデルの一般的な指示追従やCoD品質試験には代えない。

### 再現手順

patchの対象revision、licenseと適用手順は[A15 patch説明](../patches/backburner-a15/README.md)を参照する。現行checkoutや標準installerへ無条件に適用せず、実験用checkoutで扱う。

1. 同じQwen3.5 GGUFからdecoder-only copyと`-L 28 --no-head`のtailを作る。
2. APFS build directoryを指定し、`BB_A15_LITE=1 BB_INCREMENTAL_IOS=1 JOBS=2`でframeworkとappをbuildする。署名teamは自分の有効な開発teamを使う。
3. 自分の実機へappをinstallし、app containerの`Documents/tail.gguf`へtailをUSB配置する。
4. appを`PA_NA_DISABLE=1`と`GGML_METAL_FA_PREFILL_NA=0`でforeground起動する。phone controlの`mem`がtail ready、thermal0か1、memory余裕512 MiB以上、cable interface type2、Wi-Fi未pairであることを確認する。
5. 同じserverとモデルでMac単独とphone分担を別の新規出力先へ測定する。serverは同時に1つだけ実行し、既存portのlistenerを勝手に停止しない。

```sh
python3.11 tools/backburner_phone_trial.py \
  --server /path/to/patched/llama-server \
  --model /path/to/qwen35-4b-text-only.gguf \
  --mode mac --batch 512 --ubatch 128 --rounds 2 \
  --out /path/to/audits/new-layer-mac

python3.11 tools/backburner_phone_trial.py \
  --server /path/to/patched/llama-server \
  --model /path/to/qwen35-4b-text-only.gguf \
  --mode phone --phone '<wired 169.254.x.x>' --layer 28 \
  --batch 512 --ubatch 128 --rounds 2 \
  --out /path/to/audits/new-layer-phone
```

`trial.json`にraw、入力とWeightのSHA、全引数、時間、phoneの前後状態、計算とmergeの証拠、停止理由を保存する。単なる接続成功、生成の途中切れ、phone計算後のMac fallbackは成功としない。実験scriptが起動したserverだけを終了する。

## 検証と保存先

Pythonの全134単体テスト、SwiftのUUID・job範囲・streaming SHA検査、物理iPhone向けbuildとinstall、上記の実生成を確認した。互換patchは固定upstreamの未変更ファイルに対する適用確認が通過した。単体テストやbuildの成功と、実機の推論成功は別の記録として扱う。

全rawのrootは以下。

```text
/Volumes/data4/cod_model_weight/experiments/backburner-a15.jRIvD5
  independent_compare_01     初回baseline契約失敗を保存
  independent_compare_02     完了した独立job分担
  layer_mac_01               batch256のMac先行
  layer_phone_01             batch256のphone後行
  layer_phone_02_reverse     batch256のphone先行
  layer_mac_02_reverse       batch256のMac後行
  layer_mac_03_batch512      batch512のMac先行
  layer_phone_03_batch512    batch512のphone後行
  layer_phone_04_batch512_reverse
  layer_mac_04_batch512_reverse
```

現段階では実験用app・source・patchだけを残し、Base GGUF、Adapter、raw機器情報をGitへ含めない。ExtremeWeather・horse・dotsとの本番接続、完全な討論の時間・意味保持、端末切断時の運用、継続負荷とbatteryの評価は別gateとして必要である。
