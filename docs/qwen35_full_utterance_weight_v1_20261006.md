# Qwen3.5 全文発言用Weightの学習

## 現在の状態

本人の候補理由・専門観点・原文evidenceを受け取る全文utterance rendererのLoRAを、Qwen3.5-4B-4bitで128ステップ実学習した。
32・64・96・128ステップの保存Weightを作成し、開発24例でstep32を選定した。本文だけの言い換えではなく、発言のJSON・move・理由を対象にしている。
Baseの主張選択・根拠選択・投票にこのAdapterを使わない。
未学習24例はBaseの直接発言検査12/24に対しstep32が13/24で、JSON契約は双方24/24だった。
未確認の効果を断定する誤り、同意・対案の発話行為の欠落、内部D番号の露出が残り、研究HOLDである。既定Weightの置換やWeight公開はしない。

## 教材と入力

新規9学習topicの72発言、3開発topicの24発言、3未学習topicの24発言を固定した。
提案・対案・問題点のみの異議・同意・改善・補足・維持・修正を、8人格で偏りなく扱う。
学習9topicを通じて、各人格が8種類全てのmoveを含む。反対専用の人格にはしない。
同じ主張を複数人が支持する教材も含み、毎回「確かに見落としていました」とは書かない。

学習側だけでは、原文と合わない候補理由を渡す対照も追加し、144表現を用いる。
候補理由は追加の証拠ではなく、原文と選択を優先する。
確認済みの比較方向・条件範囲と、未確認の効果、数量上限、対象の除外を区別する。
現在のsource-grounding診断、台風、競馬、旧最終評価の回答を正解として取り込んでいない。

すり合わせには実runtimeと同じ形の長いIDを用い、発言本文へIDや内部キーを読み上げない。
出力は`{"utterances":[{"id":"入力id","utterance":"発言"}]}`だけ。
語尾補正、code側の接続句、代替文を、直接生成の成功へ加点しない。

実MLX ChatDatasetの192表現を検査した。思考なしのprefix、正解JSONだけのmask対象が一致した。
入力と正解を含む最大tokenは学習698・開発713・未学習715で、max sequence768以内だった。
切り捨てた入力を完全な検証として扱わない。

## 学習条件

Baseから、最後4層の`mlp.down_proj`だけをrank4・scale8で初期化した。
初期B行列は全てゼロで、同じ発言1件の生成が初期化前後で一致した。
この初期化の照合は1件のprobeであり、完成Weightの全入力非回帰とは違う。

batch1、learning rate1e-5、gradient checkpoint有効、128ステップを2499.567秒で完走し、32ごとに保存した。
32ステップでは学習loss1.270、開発loss1.116、96では開発loss0.796、128では0.985だった。MLXピークは16.003GB。
Base初期開発lossは1.559。lossの低下だけで機能改善や昇格とは扱わない。
学習速度は約0.044〜0.065 step/秒で、短い本文学習より重い。

外付けTMPDIR・オフライン指定・20GiB/2秒の容量監視で逐次実行し、同時Model推論はしない。
モデル、dataset、Adapter、監査出力は外付けに置き、Baseを内蔵へ複製しない。

## 開発選定と未学習評価

全24開発例で、JSON契約・move・凍結claim・根拠数値・反応整合の違反が少ない候補、
直接合格が多い候補、早いcheckpointの順に選んだ。選定は未学習24例の生成前に固定した。
現行runtimeと同じ検証関数を使い、parserの成功を意味の完全な証明にはしない。

| 条件 | JSON契約 | 直接発言検査 |
| --- | --- | --- |
| Base 開発 | 24/24 | 10/24 |
| step32 開発 | 24/24 | 13/24 |
| step64 開発 | 24/24 | 9/24 |
| step96 開発 | 24/24 | 13/24 |
| step128 開発 | 24/24 | 13/24 |
| Base 未学習 | 24/24 | 12/24 |
| step32 未学習 | 24/24 | 13/24 |

未学習では改善発言2件が直接合格へ変わり、維持発言1件が後退した。全体の小幅な増加は、個別非回帰や意味の正しさを保証しない。
step32の地図配布の補足は、原文の「効果は未確認」を「効果も確認済みです」と反転したが、直接発言検査には通った。
楽器確認の維持発言でも、原文にない「計測結果の独立性」「後続の再計算精度」を効果のように語った。
専門観点は議論の評価軸であり、それ自体を測定済みの根拠として扱えない。

Adapterを外した前後でBaseの発言原文1件は一致した。これは単一の決定的probeで、全入力の隔離証明ではない。
旧本文評価は保存済み22ファイル1173行の入力・原文・採点を再検査し、一致を確認した。新たな全文生成1173件ではない。

## 実討論と残る検証

旧全文18例は以前と同じ入力・structured systemを使った。正負source対照16例は、
台車と傘の架空資料について未確認／確認済みを切り替え、原文から作った本人の候補理由だけを渡した。
全34入力を条件固定後にBaseとstep32で生成し、正解文や他人格の発言は入力しなかった。

| 条件 | 旧全文JSON契約 | 旧全文直接検査 | 正負source JSON契約 | 正負source直接検査 |
| --- | --- | --- | --- | --- |
| Base | 13/18 | 10/18 | 16/16 | 16/16 |
| step32 | 3/18 | 0/18 | 16/16 | 13/16 |

旧全文では14例の`utterances`が配列でなく辞書になり、1例はJSON objectの契約自体に違反した。
残る3例にもmove検査の不合格がある。自然な異議が固定マーカーに合わない場合もあり、
この数値はruntimeとの互換性であって、文章全てが意味的に誤りだという判定ではない。
ただし、全面展開という自分の凍結主張を捨てて相手の限定pilotを選ぶ反転や、reviseを維持に変える誤りも残った。

正負sourceでは、Baseが「電動4日」を「手押し2日」より工数で有利と誤読した。
step32はこの誤読を同じ話者の発言から除いたが、確認済みの追加費用0円も4人格全員で省略した。
難しい事実を言わなくなったことを、正しい根拠理解の改善とは数えない。
傘の確認済み対照では雨が出発5分後から45分後、外出が18分なのに、step32の2発言は「18分間雨が降る」と期間を取り違えた。
雨具を持たない選択を、折りたたみ傘の保護効果で正当化する理由の混線もある。

8人格の同一入力比較は、既存の`--renderer-adapter`経路で実行する。
通常のAdapterなしBaseは3発言ずつ、Adapter付きは1発言ずつになるため、
対照にはB行列がゼロのAdapterを使用し、双方1発言ずつの条件に揃える。
これは「Base＋ゼロ効果LoRA」の対照であり、通常の3発言バッチBaseとの速度差をWeightの効果とは扱わない。
既知の2案fixtureの確認と、自由な新規多案議論の証明は区別する。

8人格の比較は双方40 callで完走した。独立判断8件・投票8件の原文、rendererの24入力、結論は一致した。
各判断のAdapterは`null`で、各発言に本人の候補理由と設定された観点だけを渡した。別人格の原文を入力していない。

| 条件 | 直接生成 | 補正済みモデル発言 | 代替発言 | 完全JSON契約 | 実行時間 |
| --- | --- | --- | --- | --- | --- |
| Baseとゼロ効果LoRA | 0/24 | 12/24 | 12/24 | 23/24 | 474.553秒 |
| step32 | 0/24 | 12/24 | 12/24 | 24/24 | 489.655秒 |

双方とも討論全体のhard gateは不合格である。この1比較から高速化は確認できない。
すり合わせは1ラウンドで8人がSwift先行を選び、変更投票は0件だった。判断変更を伴う自然な対話の検証にはならない。
BaseにはSwift向け未算定工数をRustの再利用可能性へ付け替える誤り、否定の「移植不要とはしていない」を「移植不要」へ反転する誤りが残った。
step32の発言にも、試作2週間を初版完成2週間へ変える誤りや、3人格がほぼ同じラベルだけを述べる例がある。
自動の近似重複率0は、これらの似た言い回しや独立した理由の不足を証明しない。

残る改善は、入力形式が変わってもJSON契約を維持すること、確認済み／未確認・対象・比較方向・試作と完成を保つこと、
同意する人格にも別々の根拠に即した理由を生成させることである。
旧本文Weightの基準、凍結raw、source-grounding診断の答えを緩めて候補を通さない。
Generalの自由な議論やBaseの誤読改善まで、この学習のlossだけで完成扱いにしない。

## 発言の切り詰め修正

比較完了後、共通の表示処理が「根拠」以降を一律に捨てる不具合を修正した。
モデル原文の「この提案は根拠として十分ですが、工数の見積もり範囲に注意が必要です。」が、
補正後に「この提案は。」へ変わっていた。現在は、末尾の独立した「根拠は[D01]です。」などの注記だけを除去する。
通常の説明文は残し、fallbackとmove補正で同じ処理を使う。

この修正は実保存入力1件の再処理と107単体テストで確認した。全文168行と過去本文1173行の直接採点は不変だった。
上表の実討論は修正前の凍結コードによる結果で、表示修正後の新たなnative討論を実行したとは扱わない。
修正前のコードSHAは`6238a410dbb3ef2c52840d751cddff7a1502f373a41ffbca13110cd564fe3cea`で、外付けの`source_snapshot/cod_model.py`へ保存している。
表示の改善を、学習Weightの精度改善へ加点しない。

## 再現と保存先

```sh
<mlx-python> tools/general_utterance_training.py build \
  --curated data/general_utterance_qwen35_v1/curated.json --out /path/to/new-dataset

<mlx-python> tools/general_utterance_training.py train \
  --model /path/to/Qwen3.5-4B-4bit --data /path/to/new-dataset \
  --config configs/general-utterance-qwen35-4b-v1.yaml \
  --parent-adapter /path/to/zero/adapters.safetensors --out /path/to/new-adapter

<mlx-python> tools/general_utterance_training.py evaluate \
  --model /path/to/Qwen3.5-4B-4bit --adapter /path/to/new-adapter \
  --split valid --out /path/to/new-evaluation.json
```

```text
dataset: /Volumes/data4/cod_model_weight/datasets/general-utterance-qwen35-4b-v1/mlx_runtime_ids
adapters: /Volumes/data4/cod_model_weight/adapters/general-utterance-qwen35-4b-v1
audit: /Volumes/data4/cod_model_weight/evaluations/general-utterance-qwen35-4b-v1_20261006
```

`plan.json`は学習・最終生成前の条件、`prepare.json`はゼロ効果の初期化、
`final_token_audit.json`は実入力のmask検査、`selection.json`は最終評価前に固定した開発選定、
`base_fresh24.json`と`selected_fresh24.json`は未学習題材の同一入力生成である。
`gates_plan.json`・`gates_inputs.json`は旧全文と正負source、実討論の固定条件で、
`fresh_manual_review.json`・`controls_manual_review.json`は元rawを編集しない意味の確認記録である。
`gates_comparison.json`に同一入力とAdapter分離の照合、`runtime_manual_review.json`に公開48発言の意味と由来、
`display_fix_audit.json`に切り詰め修正前後の実入力再処理を保存した。
`runtime_base`と`runtime_selected`の`event_debate_*.json`には、モデルの原文、本人の候補理由、補正・代替後の会話を別々に保持している。
選定Weightは754632 bytes、SHA256 `530d07dfd66436c121e76ae6b31d5a0ab7a56cc22b8a80ba096dbba0b72e54e4`。
Baseが別途必要なLoRAであり、単体で使える軽量Baseモデルではない。
新しい[教材](../data/general_utterance_qwen35_v1/curated.json)、[設定](../configs/general-utterance-qwen35-4b-v1.yaml)、
[学習CLI](../tools/general_utterance_training.py)を使用する。
