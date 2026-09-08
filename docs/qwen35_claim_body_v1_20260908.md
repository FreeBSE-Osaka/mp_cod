# Qwen3.5-4B Claim Body LoRA v1 — 2026-09-08

## 結論

Qwen3.5-4B用の本文LoRAを新規に学習し、保存されたstep32 / step64を比較した。
選定はstep32（4,065,793 bytes、約4.07MB）。**研究用HOLDで、既定Weightは置き換えない。**
既存15問はBase 1/15から15/15へ改善したが、新しい12問の直接合格は6/12から7/12に留まり、
利用制限の意味を変える例も残る。これは8人格の判断そのものを学習したWeightではなく、共有の本文renderer。

## 検証範囲

Qwen3.5-4Bの既存MLX 4bitモデルから新しい本文専用LoRAを学習する。
Qwen3-1.7BのAdapterも、以前のQwen3.5全文renderer step20も引き継がない。
既存の学習・評価ツールを再利用し、Baseが主張・証拠・投票を決め、LoRAが本文だけを生成する分離を保つ。
9B、GGUF変換、TensorSharp導入、iPhone実機動作は今回の検証対象ではない。

## 学習の中断

96 iterationを予定したが、88の進捗報告後に以下のエラーで終了した。
正常に保存されたstep32 / step64を評価対象にし、存在しないstep96を選定に含めない。
変更理由は最終holdout生成前に`audit/training_interruption.json`へ記録した。

```text
RuntimeError: [METAL] Command buffer execution failed: Internal Error (0000000e:Internal Error).
```

validation lossは開始0.616、step32で0.200、step64で0.108。
報告peak memoryは7.521 GB。メモリ不足や温度制限が原因とは確定していない。
別の推論・シミュレーターも稼働しており、他のユーザープロセスは停止していない。

終了後の評価はモデルの読み込みに長く待機した。
`sample`で`ParallelFileReader::read` / `pread`を確認し、GPU計算待ちとは区別した。
外付けdata4はUSB / UFSD_NTFS。比較を続けるため同じモデルを内蔵の作業用キャッシュへコピーした。
学習時のMetalエラーと、このファイル読み込み待機を同一原因だとは断定しない。

## 凍結条件

- Base: `/Volumes/data4/cod_model_weight/models/Qwen3.5-4B-4bit`
- 設定: `configs/claim-body-qwen35-4b-v1.yaml`
- 学習/推論の同一system: `configs/claim-body-qwen35-4b-v1-system.txt`
- 学習計画: 96 iteration、batch 1、最後の4層、rank 4、scale 8、learning rate 1e-4。実際は上記のとおり中断。
- 更新対象: 1.015M / 4205.750M parameters（0.024%）。Base本体を更新しない。
- データ: v3 rehearsal + Generalの手作成事例、train 945 / valid 91 / test 123。
- 入力込み最大token: train 261 / valid 265 / test 251。設定384以内で切り詰めなし。
- Qwen3.5 tokenizerで学習テキストの先頭が推論prompt（`enable_thinking=False`）と一致することを確認。
- 選定: step32 / 64 / 96を開発用8問で比較。直接合格、語尾補正込み、早いstepの順で選ぶ。
- 選定後: 今回新しく固定した12問（8話者・3topic）と既存15問でBaseと比較。
- 新規holdoutは学習データとclaim重複なし。model生成結果を見る前に教材・選定規則を固定。

新規holdoutは [`data/general_body_qwen35_v1_holdout/curated.json`](../data/general_body_qwen35_v1_holdout/curated.json)。
community_garden / lost_property / reading_sessionを含む架空の条件確認であり、現実の規則や予測ではない。
前回見たGeneralのtestを新規の盲検試験とは呼ばず、別問題を用意した。
新規holdoutの正解文は訓練に使わない。

## 判定について

厳格JSON、丁寧完全文、claim整合、数値、必須条件、競合claimの不選択を確認する。
語尾修復で通った結果と、モデルが直接通った結果は分ける。
検査器は意味理解の完全な証明ではないので、rawの目視監査も行う。
否定・上限・時制の崩れを見つけた場合、合格数を上げるために検査を緩めない。

## 開発用8問の結果と選定

| モデル | 直接合格 | 語尾補正込み | strict JSON |
|---|---:|---:|---:|
| Qwen3.5-4B Base | 1/8 | 1/8 | 8/8 |
| 新規LoRA step32 | 5/8 | 6/8 | 8/8 |
| 新規LoRA step64 | 5/8 | 5/8 | 8/8 |

事前基準に従いstep32を選定し、最終holdout生成前に`audit/selection.json`へ保存した。
Baseは新しい実行でも前回と同じ1/8だった。
step64は「金曜日だけにとし」「返けし」のような不自然な接続や誤字を生成した。
step32では同じ箇所が「金曜日だけに限定し」「返却し」だった。

不合格には、正しい同義表現が必須アンカーに一致しない例もある。
例えば「予約資料の受取には使用しません」は「使いません」と意味は同じだが、この凍結検査では不合格。
「購入を確約するものではないです」も強い意味の反転とは扱わない。
直接合格率を、モデル全般の能力や完全な意味正確率と解釈してはいけない。
学習後にアンカーを緩めて合格数を増やす変更はしていない。

## 最終holdout結果

候補固定後に、新しい12問と既存15問をBase / step32で一度ずつ生成した。
全体では8話者を含む。いずれもstrict JSONは27/27。
以下は保存rawへ同じ補強後の検査を適用した結果で、再生成・再学習は行っていない。

| モデル | 新規12問・直接 | 新規12問・語尾補正込み | 既存15問・直接 | 既存15問・語尾補正込み |
|---|---:|---:|---:|---:|
| Qwen3.5-4B Base | 6/12 | 6/12 | 1/15 | 1/15 |
| 本文LoRA step32 | 7/12 | 9/12 | 15/15 | 15/15 |

原始集計ではBaseが新規8/12・既存3/15だった。しかし、新規で「含めるです」「出すです」を、
既存で18%→18割 / 11%→11割への誤変換を検査が見逃した。
`cod_model.py`の既存の語尾検査と割合単位検査を補強し、両モデルのrawを再採点した。
%とパーセントの表記差は許可するが、%と割のすり替えや割合の符号変更は拒否する。
これは明示的な割合表記の検査で、全ての単位・対象と数値の対応・意味の正しさを保証するものではない。

語尾補正の追加で、step32の「仮の案内だけを出す。」は「出します。」へ補正可能になり、
補正込み合格は8/12から9/12へ増えた。この1件はコード側の改善であり、直接のWeight改善には数えない。
開発用8問も同じ補強後検査で再採点し、選定順位・合格数が変わらないことを確認した。

## 生の生成例と残る問題

以下は本文単体テストの`body`原文。複数人が実際に討論した実況ではない。

```text
仮説構築者: 水やり当番は午前8時から9時までで、薬剤散布は担当しません。
長期影響評価者: 本人確認ができない場合は返却を保留します。
利用者体験研究者: 読み聞かせは1回25分までとし、質問の時間も含めます。
```

新規テーマでの主な失敗:

- 元の主張「共同区画の利用は登録者に限り見学者には道具を貸し出さない」に対して、
  「共同区画の利用では、登録者には道具を貸し出しますが、見学者には貸し出しません。」と生成。
  区画の利用制限が抜け、登録者への貸出を積極的に保証する内容へ変わったため不合格。
- 「住所を伝えないです」「14時は変えないです」は凍結した丁寧語アンカーに不一致。
  この2件を明確な意味反転と扱うわけではなく、表現検査の保守的な不合格として区別する。
- 「認める。」「出す。」と常体で終わるものは、直接合格に含めない。意味保持を確認した語尾補正は別集計。

既存15問には同じclaimを複数話者へ渡したケースが含まれるため、15の独立したテーマへの汎化証明ではない。
新規12問も3topicの小規模な合成試験であり、通常の自由討論全体の精度を示すものではない。

## Adapter分離と回帰確認

- `--check-adapter-isolation`で、同じ本文1問のBase出力がAdapter着脱の前後で完全一致した。
- 本体Weightと既存Adapterの置換は行っていない。
- `python3.11 -m unittest -q`: 49テスト成功。8話者holdout、割合単位・符号、語尾、分離オプションの検証を含む。
- 既存のA15台風保存JSONをPython validatorで再確認しvalid。
- Swift importerへの新しい割合guard移植、Qwen3.5のiPhone実走、8人のCoD実走は今回未実施。

新しいholdoutでの意味保持と実際の討論経路が未達/未検証なので、既定採用やWeight公開へ進めない。

## 成果物の保存先

```text
dataset: /Volumes/data4/cod_model_weight/datasets/claim-body-qwen35-4b-v1/mlx_shared
adapter: /Volumes/data4/cod_model_weight/adapters/claim-body-qwen35-4b-v1
audit:   /Volumes/data4/cod_model_weight/evaluations/claim-body-qwen35-4b-v1
```

`audit/evaluation_plan.json`には、学習・データ・新規holdoutのSHAと選定規則を保存している。
`audit/train.log`は学習プロセスの出力。既存のWeightと前回の記録は上書きしていない。
`audit/model_identity.json`は元モデルと作業用コピーの対応を記録する。
モデル・tokenizer・設定の10ファイルは`diff -rq`で一致を確認した。
元ディレクトリだけに存在するダウンロード用`.cache`と`.gitattributes`はコピー対象外。
一時キャッシュは検証後に削除し、元の外付けモデルを保持する。rawにあるキャッシュpathは当時の実行履歴であり、
再実行時は`model_identity.json`のsourceとSHAを参照する。

```text
Base model bytes:   3034300695
Base model SHA256:  5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db
Adapter bytes:      4065793
step32 SHA256:      be663bcb69fde2161061bf4a2b6f02b3e6b687b11e7c4f1b3bd87aa6c9ad7b04
step64 SHA256:      4b30eff0f306ac1f63d6ea94b9ecad07e0d21d106b0f727e5b4f253cbd9a9a38
Adapter config SHA: 4c891bd0ae11b30776e06460ddf8be2a2ad93aae657f83628b101e6e774fb741
```

監査rootの主な記録:

- `step32_dev.json` / `step64_dev.json` / `base_dev.json`: 開発用生成raw。
- `selection.json`: 最終生成前の候補固定。
- `base_final.json` / `step32_final.json`: 新規12問＋既存15問の全raw、後者にはAdapter分離チェックも保存。
- `*_guard_v2.json`: 保存rawの共通検査による再採点。元のrawファイルは変更していない。
- `training_interruption.json` / `train.log`: 96未完走の理由と正常保存checkpoint。
- `stalled_inference_sample.txt`: モデル読み込み待機のサンプル。Metalエラーの原因を証明するものではない。

配布判断は [`promotions/qwen3.5-4b-claim-body-v1-step32.json`](../promotions/qwen3.5-4b-claim-body-v1-step32.json) にも保存する。

## 再現手順

以下の出力先は未使用のディレクトリ/ファイルにする。MLX用Python環境が必要。

```sh
python3.11 tools/general_body_training.py build \
  --rehearsal /path/to/claim-body-v3/mlx_shared \
  --renderer-system-file configs/claim-body-qwen35-4b-v1-system.txt \
  --out /path/to/new-dataset

<mlx-python> -m mlx_lm lora --train \
  --model /path/to/Qwen3.5-4B-4bit --data /path/to/new-dataset \
  --config configs/claim-body-qwen35-4b-v1.yaml --adapter-path /path/to/new-adapter

<mlx-python> tools/general_body_training.py evaluate \
  --model /path/to/Qwen3.5-4B-4bit --adapter /path/to/checkpoint \
  --split valid --out /path/to/dev.json

<mlx-python> tools/general_body_training.py evaluate \
  --model /path/to/Qwen3.5-4B-4bit --adapter /path/to/selected-checkpoint --check-adapter-isolation \
  --curated data/general_body_qwen35_v1_holdout/curated.json --split test \
  --legacy /path/to/contract_v2_external_15.json --out /path/to/final.json
```

比較するBaseでは`--adapter`を省略し、同じprompt・入力・最大160 tokens・temperature 0を使う。
その際`--check-adapter-isolation`も省略する。分離チェックは同じ本文1問で、Adapter着脱前後の
Base出力の一致を確認するものであり、全ての構造判断の非回帰証明ではない。
