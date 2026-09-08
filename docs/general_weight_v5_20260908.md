# General Claim Body v5 / v5b — 2026-09-08

## 結論

8人の専門観点、同じ意見の複数人支持、対案を強制しない反論、賛同と改善を実装し、
General本文専用LoRAを2条件で実際に学習した。候補v5b step256は**研究用HOLD**。
Weightは存在するが、親v3の既定置換・自動昇格・Weight公開は行わない。

新規未学習12問の直接合格は親の4件から7件へ増えた一方、語尾補正込みでは親9件・候補8件。
否定や上限の脱落、提案の過去形化が残った。loss低下だけを成功としない。
今回の変更は既存のBase判断と本文rendererの分離を保ち、新しい推論エンジンは追加していない。

## 学習したもの

- Base: `mlx-community/Qwen3-1.7B-4bit` の既存ローカルWeight。
- 初期Adapter: Claim Body v3 `shared_step128`。v5bもv3から開始し、v5の続きではない。
- 共有の本文LoRA: 最後の4層 / rank 4 / scale 8 / batch 1 / 最大704 tokens。
- claim選択・D番号・投票はBaseが担当。LoRAに判断権を移していない。
- 8人格別に8個のAdapterを作ったわけではなく、8話者を含む共有Adapter。
- 環境: M2 / 24 GiB、Python 3.11.15、MLX 0.32.1、MLX-LM 0.31.3、Transformers 5.15.1。

| 条件 | v5 | v5b |
|---|---:|---:|
| iteration | 192 | 256 |
| learning rate | 1e-5 | 2e-5 |
| gradient accumulation | 4 | 1 |
| validation範囲 | 全91件 | 各回16 batch |
| 同じrun内のvalidation loss | 0.278 → 0.155 | 0.193 → 0.088 |
| MLX報告peak memory | 1.700 GB | 1.700 GB |

validation範囲が異なるため、2列のlossをそのまま優劣比較してはいけない。
v5の直接生成は十分改善せず、v5bのstep128と256を開発用8問で比較して256を選定した。
最終test生成より先に選定を `selection_dev.json` へ保存した。

## データと凍結

新規データは手作成の架空事例。討論の自動生成ログを無審査で学習に戻していない。
新規trainは9 topic・45事例、validは別の2 topic・8事例、testはさらに別の3 topic・12事例。
話者名を8種類に展開し、従来v3のrehearsalデータを加えた。

| split | v3 rehearsal | 新規（話者展開後） | 合計 | 入力込み最大token |
|---|---:|---:|---:|---:|
| train | 585 | 360 | 945 | 291 |
| valid | 27 | 64 | 91 | 295 |
| test | 27 | 96 | 123 | 277 |

64/96件を独立な事例数とは数えない。内容の独立評価は8/12問。
新規testのtopicはmuseum_booking、neighborhood_shuttle、equipment_lending。
新規train/valid/test間で同一claimを共有せず、topic単位で分離した。
token数はtokenizerの `return_dict=False` で数え、全件704以内を確認した。

学習時systemは [`configs/claim-body-v5-system.txt`](../configs/claim-body-v5-system.txt) に固定。
その後runtimeへ「です・ます調」を明示する一文を追加したため、学習時と評価時のsystemは異なる。
最終比較は全モデルに同じ評価systemを使用し、prompt改善をWeightだけの効果とは扱わない。

## 同一条件の生成比較

温度0、`enable_thinking=False`、最大160 tokens、固定入力で生成。
直接合格はstrict JSON・完全文・丁寧語・claim整合・数値・必須条件の保持・競合claim不選択を全て要求。
語尾補正込みは別列にし、検証済みstatementへのfallbackは合格に含めない。
これらは保守的な自動検査であり、人間による完全な意味・自然さの採点ではない。

### 開発用8問（最新guardで保存rawを再採点）

| モデル | 直接合格 | 語尾補正込み | strict JSON | 生成合計秒 |
|---|---:|---:|---:|---:|
| Qwen3-1.7B Base | 0/8 | 3/8 | 4/8 | 18.84 |
| 親v3 step128 | 1/8 | 4/8 | 8/8 | 10.98 |
| v5b step256 | 5/8 | 7/8 | 8/8 | 10.81 |
| Qwen3.5-4B Base | 1/8 | 1/8 | 8/8 | 44.49 |

時間はロードを除く各生成呼び出しの合計。端末上で別処理も動いており、厳密な速度benchmarkではない。
Qwen3.5-4Bは既存のMLX 4bitモデルを今回追加確認した。JSONは揃ったが「戻すです」「伝えるです」を
当初の語尾検査が誤って合格させたため、全モデル共通のguardを補強した。
これは本文専用契約への適合結果であり、Qwen3.5全体の能力や9B版の評価ではない。

### 最終holdout（最新guardで保存rawを再採点）

| モデル | 新規12問・直接 | 新規12問・語尾補正込み | 既存15問・直接 | 既存15問・語尾補正込み |
|---|---:|---:|---:|---:|
| Qwen3-1.7B Base | 0/12 | 6/12 | 0/15 | 2/15 |
| 親v3 step128 | 4/12 | 9/12 | 13/15 | 13/15 |
| v5b step256 | 7/12 | 8/12 | 13/15 | 13/15 |

親と候補は27/27 strict JSON。意味まで27/27正しいという意味ではない。
生成時の旧検査では既存15問が親14/15、候補15/15だったが、時制・話者混入等の追加検査で両方13/15になった。
rawは変更せず、`*_guard_v2.json` に再採点結果を別保存した。
語句アンカーの「再度確認→再確認」等の限定的な同義処理はtest生成前に確定し、test結果を見て緩和していない。

不合格例:

- 「動作確認が済むまで貸し出さない」が「済ましませんまで…貸し出します」に変化。文法崩れと否定反転。
- 「最長9日」から「最長」が脱落。上限を固定期間へ変えてしまう。
- 提案「傘を持って出る」が「傘を持って出しました」に変化。未実行の行動を完了扱いする。
- アンカーが同義表現を拾えない保守的な不合格もあるが、それだけでなく上記の実質的誤りもある。

## 8人の実走と発言

架空の傘相談で8人、19 event、1回のすり合わせ8票を実行。
全役が全8 claimを選択可能（並び順で関心を表す）。同意を減点せず、意見の不足をコードで埋めない。
19件は全てBaseが選んだ有効claimで、補完claimは0。別の1件は不正なD番号の組合せで拒否。
短い `LEFT/RIGHT` を人格ごとに左右反転して投票し、ログへ元のclaim codeと対応表を保存した。

最終票は傘を置く5・持つ2・両論保持1。8人の閾値6に届かないため未解決を保持。
32 model call・309.955秒。初期8 claim呼び出しと再投票の修復が重く、低遅延用途にはまだ不向き。
本文cacheは18 hitだったが、同じclaimでは同じ文が繰り返されやすい。

以下は公開表示ログの抜粋。本文はLoRA生成で、記録された安全な語尾補正を含む。
「私も同じ見方です」等の導入句はコード合成であり、全文章をモデルが生成したわけではない。

```text
仮説構築者: 別の進め方として、外出が15分と短いので濡れるリスクを受け入れて傘を置きます。
批判的設計者: ただ、気になる点があります。降り出す時刻が分からないので、短時間でも濡れないとは言えません。
資源・制約設計者: 私も同じ見方です。濡れる不便を避けたいという希望も傘を持つ理由になります。
利用者体験研究者: そこは同意します。濡れる不便を避けたいという希望も傘を持つ理由になります。
利用者体験研究者: その案に賛成です。荷物の軽さを優先するなら、傘を置く選択にも理由があります。
実証監査者: 出発直前に空の様子と予報を再確認して持ち物を決めます。
```

実走全体はHOLD。生成時には本文26/27を採用したが、後の目視監査で過去形化と、入力にない
「実証監査者にとって…災害時の備え」を付け足す本文を発見した。
最新guardで同じ9生成batchを再検査すると、語尾補正込みで6/9。拒否された3 batchはcache込み9発言に対応する。
この再検査は保存rawの確認であり、修正後に8人討論を再生成して完走したという証拠ではない。

同じ意見を複数人が持てる構造は動作したが、人間同士のような自在な対話や全発言の意味安全性は未達。
固定台帳外の新規意見も出せる通常 `debate` は別経路であり、この証拠検証済みイベント実走と混同しない。

## Qwen3.5 / TensorSharp / iPhoneについて

開発者自身がiPhone 17 Pro MaxでQwen3.5 9B IQ4_XSをTensorSharp上で動かしたと報告している。
モデル推論・コード実行はローカルだが、株価の取得には通信を使用すると明記している。
端末内推論と全作業オフラインを区別し、通信やツール権限の確認なしに完全なプライバシー保証とは扱わない。
[開発者の投稿](https://www.reddit.com/r/Qwen_AI/comments/1w8708m/running_qwen35_9b_as_a_fully_local_ai_agent_on_a/)

TensorSharpはGGUFを扱う実行エンジンで、対応する量子化演算を備えるため、この経路でGGUFをMLX形式へ
変換し直す必要はない。GPU/MetalとNPUは別物であり、このデモのNPU利用・6倍速という条件は確認できていない。
[公式リポジトリ](https://github.com/zhongkaifu/TensorSharp)

CoDの構造判断側をQwen3.5へ替えることは可能。一方、今のQwen3-1.7B用LoRAはQwen3.5用ではないので
そのまま流用せず、対応Baseで別に学習・評価する。Qwen3.5にはGated DeltaNetを含む異なる構成がある。
[Qwen公式モデルカード](https://huggingface.co/Qwen/Qwen3.5-9B)

今回は9Bのダウンロード、TensorSharp導入、iPhone 17での再現、既存iPhoneアプリの変更はしていない。
Macの4B結果もiPhoneの速度やメモリ上限を保証しない。既存A15の経路は別に維持する。

## 再現・成果物

ソースの [学習・評価ツール](../tools/general_body_training.py)、[v5設定](../configs/claim-body-v5.yaml)、
[v5b設定](../configs/claim-body-v5b.yaml)、[新規事例](../data/general_body_v5/curated.json) を使用。
以下の `/path/to/...` は自分の環境に合わせる。既存の凍結データ/Weightは上書きせず、新しい出力先を使う。

```sh
python3.11 tools/general_body_training.py build \
  --rehearsal /path/to/claim-body-v3/mlx_shared --out /path/to/new-dataset

<mlx-python> -m mlx_lm lora --train \
  --model /path/to/Qwen3-1.7B-4bit --data /path/to/new-dataset \
  --resume-adapter-file /path/to/claim-body-v3-step128/adapters.safetensors \
  --config configs/claim-body-v5b.yaml --adapter-path /path/to/new-adapter

<mlx-python> tools/general_body_training.py evaluate \
  --model /path/to/Qwen3-1.7B-4bit --adapter /path/to/new-adapter \
  --split valid --out /path/to/new-dev-evaluation.json

python3.11 tools/general_body_training.py rescore \
  --input /path/to/saved-evaluation.json \
  --legacy /path/to/contract_v2_external_15.json --out /path/to/new-rescore.json
```

凍結rehearsalと親Adapterは外付け側の既存成果物が必要。新規事例だけで同じ学習を再現できるわけではない。
開発用で候補を固定してから別topicの `--split test` と既存 `--legacy` を生成する。
`rescore` は再生成も再学習もせず、元ファイルを上書きしない。

ローカル成果物のルート:

```text
datasets: /Volumes/data4/cod_model_weight/datasets/claim-body-v5-general/mlx_shared
v5:       /Volumes/data4/cod_model_weight/adapters/claim-body-v5-general
v5b:      /Volumes/data4/cod_model_weight/adapters/claim-body-v5b-general
selected: /Volumes/data4/cod_model_weight/adapters/claim-body-v5b-general/step256
audit:    /Volumes/data4/cod_model_weight/evaluations/claim-body-v5-general
```

監査root内の主なファイル:

- `selection_dev.json`: test生成前の候補選定。
- `parent_final_holdouts.json` / `v5b256_final_holdouts.json` / `base_final_holdouts.json`: 完全な生成raw。
- 同名の `*_guard_v2.json`: 最新検査の同条件再採点。
- `qwen35_4b_base_dev.json` / `qwen35_4b_base_dev_guard_v2.json`: 4B Base追加確認。
- `umbrella_final_v5b256/event_debate_20260908_101306_981035.json`: 全8人の発言、Base raw、renderer raw、投票。
- `umbrella_final_saved_body_guard_v2.json`: 保存された本文9 batchの追加検査。
- `logs/`: 2条件の学習ログと各段階の傘相談実況ログ。

選定Weightは2,496,303 bytes（約2.50 MB）。Base本体や実行時メモリを含むサイズではない。

```text
Weight SHA256:          45295709f4ee7eabc0dae5e4729f6fe6e62231c69d91998b11a47852e3e8f1ff
Adapter config SHA256:  754c168585e95a56ec7761db3c3b2ea1f473bba4c687eea8f5fb6fc995c60087
Training config SHA256: c32b852bbc61f65b56aeeb2013dfac022a10dec733b0842a684af1a537e06040
Curated SHA256:         6d2911c20b9c23374534e553d3e9698a8d7dc66767ca82d5e0868451e4548379
Dataset manifest SHA256:2c9a107eb44b06c426ba776753905521474a313a3da204daef19fcccfc62d0c0
```

判断境界は [候補記録](../promotions/qwen3-1.7b-general-v5b-step256.json) にも保存する。

## 変更後の検証

- `python3.11 -m unittest -q`: 46テスト成功。同じclaimへの複数人同意を減点しないこと、
  疎なモデル意見を水増ししないこと、左右投票の復元、意味・語尾検査、raw再採点の非上書きを含む。
- 凍結systemを使った再生成でtrain/valid/test/manifestの4ファイル全てが学習時SHAと一致。
- 既存台風A15保存JSONを `tools/validate_iphone_native_cod.py` で再検証しvalid。
- `git diff --check` 成功。新Weightの昇格テストや新しいiPhone実走の成功とは扱わない。

## 次の品質条件

次の学習は既に見たtestを正解例として混ぜず、別の教材・新しい凍結holdoutで否定・上限・時制の保持を再検証する。
native Swift側は今回の柔軟なmoveと全ての追加guardに同期していない。既存importerにはstructuredモードを使い、
Pythonの模擬テストや保存JSONの検証を新しいiPhone実走の証拠とは扱わない。
