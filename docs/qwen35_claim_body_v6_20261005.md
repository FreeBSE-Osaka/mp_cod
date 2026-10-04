# Qwen3.5-4B Claim Body v6 補助情報と学習検証

## 検証状況

v5で残った提案の完了形化と数量上限の欠落に対し、本文入力へ原文由来の補助情報を追加する方式を検証する。
128ステップを763.161秒で正常完走し、MLX報告peak memoryは7.063GBだった。
開発用でstep64を選定し、最終検証前に固定した。新規16問と8人討論の本文は改善したが、
旧集合の非回帰に失敗したため、全体昇格は見送る。研究Weightとして保持する。
研究用HOLDであり、既定prompt・既定Weightを置き換えない。

## 補助情報の内容と境界

`body_input_item`が、元のid・speaker・claimに任意の`rendering_hints`を追加する。
`claim_kind`は台帳または新規教材が明示した値だけを使う。
kindがない既存教材や台帳は`unspecified`とし、文面から提案・事実を推測して登録しない。
`proposal`は未実施の提案として扱い、他のkindは原文の時制・確実性を維持する。

`quantity_bounds`は既存の数量条件検査が原文から取り出した厳密な値・単位・比較演算子である。
正解文、学習target、テストの必須アンカー、モデルの反省文を補助情報へ入れない。
比較の補助情報は主語や条件文全体の意味を理解するものではなく、元のclaimも常に渡す。
モデルが原文の一部を省略することを許可する機能ではない。

例として、元の「予算は1900円までで人数は最低7名」からは、
1900円以下・7人以上の比較を記録する。表記は原文を保ち、比較を逆転させない。
値はFractionの文字列で保持し、丸めや新しい計算結果を加えない。

実討論では`--body-constraint-hints`、学習dataset作成・単体評価では`--constraint-hints`で明示する。
既定はOFFで、従来の`items`入力形は変えない。
実討論の補助情報は本文Adapterにだけ渡り、Baseの主張・根拠・投票promptを変更しない。
同じlabelでもkindが違う場合は本文cacheを分離する。

## 学習前の比較

同じ親v3 step64、同じ新規dev8、temperature 0で3条件を生成した。

| 条件 | 直接合格 | strict JSON |
|---|---:|---:|
| 従来systemと通常入力 | 6/8 | 8/8 |
| 新systemと通常入力 | 5/8 | 8/8 |
| 新systemと補助情報 | 5/8 | 8/8 |

補助情報だけの精度向上は確認できなかった。
新systemのみの不合格には、確認済みの過去を現在形へ変える例や、常体語尾で終える例がある。
補助情報を加えても、比較条件の「場合だけ」を省く例が残った。
新systemだけで下げた基準との比較を、全体性能の向上と扱わない。

## 教材と学習

新規の架空教材は保管、受付、清掃、練習の32件。
長い複数動作の提案、金額・時間上限、人数下限、厳密な比較境界、未確認の効果と確認済み事実を含む。
8話者へ展開した256例と、親v3のtrain全512例を混ぜ、学習は768例。
新規dev8と最終test16は別topicで、両方に8話者を含み、旧corpusのclaimとは完全一致しない。
今回も以前の最終検証の答えをそのまま新しい教材へ戻していない。

学習・評価・実討論は同じ入力関数を使う。
旧復習512例のkindは未指定のまま、新規教材のkindだけを明示する。
SFT全1,558例を実MLXラッパーで確認し、入力境界が思考なしの推論と一致し、
学習対象はbodies JSONから始まる。最大入力tokenはtrain 321・valid 316・test 320で、設定640以内。

親はv3 step64。最後4層の`mlp.down_proj`、rank 4、scale 8、batch 1、learning rate 1e-5、
128ステップ、gradient checkpoint有効、seed 20261005で追加学習する。
32・64・96・128をdevの重大違反、直接合格、早いstepの順で選び、最終生成前に固定する。
lossだけで選ばず、最終結果を見て別stepへ差し替えない。

## 再現と検証条件

```sh
<mlx-python> tools/general_body_training.py build \
  --curated data/general_body_qwen35_v6/curated.json \
  --renderer-system-file configs/claim-body-qwen35-4b-v6-system.txt \
  --rehearsal /path/to/v3/mlx_shared --constraint-hints \
  --out /path/to/new-dataset

<mlx-python> tools/general_body_training.py train \
  --model /path/to/Qwen3.5-4B-4bit --data /path/to/new-dataset \
  --config configs/claim-body-qwen35-4b-v6.yaml \
  --parent-adapter /path/to/v3/step64/adapters.safetensors \
  --out /path/to/new-adapter

<mlx-python> tools/general_body_training.py evaluate \
  --model /path/to/Qwen3.5-4B-4bit --adapter /path/to/checkpoint \
  --renderer-system-file configs/claim-body-qwen35-4b-v6-system.txt \
  --curated data/general_body_qwen35_v6/curated.json --split valid \
  --constraint-hints --out /path/to/new-dev.json

<mlx-python> cod_model.py event-debate --domain general --backend mlx \
  --model-path /path/to/Qwen3.5-4B-4bit \
  --ledger data/general_language_choice/claim_ledger.json \
  --body-adapter /path/to/selected-adapter --body-cache-scope speaker \
  --body-system-file configs/claim-body-qwen35-4b-v6-system.txt \
  --body-constraint-hints --portable-context --out /path/to/new-run
```

この指定は実験用であり、補助情報の追加だけでWeight生成の失敗を修正済みとは扱わない。
本文の出力契約、条件・数値・時制・確実性のguard、無効文の代替表示は維持する。
コードで補正した文や凍結claim由来の代替文は、Weightの直接生成成功に数えない。
新規16問・既存の全非回帰・8人実討論を確認するまで、既定への昇格やWeight公開をしない。

容量監視は既定20GiB・2秒間隔。今回もBaseの内蔵コピーを作らず、
`TMPDIR=/Volumes/data4/cod_model_weight/tmp`とオフライン指定を付けて実行した。
CLI jobの正常終了と、意味保持・自由な討論の成功は別に確認する。

## 学習後の開発用生成

step32は7/8、step64・96・128は8/8となった。従来方式の親6/8と、新方式の親5/8を上回っている。
step64では「場合だけ」、下限条件、未確認・確認済み状態、丁寧語の語尾をdev8の全件で保持した。
重大違反・直接合格・早いstepの事前順序で64を選び、未使用問題の生成前に保存した。
未使用問題・実討論はまだ未完了で、dev全件合格を一般性能向上とは扱わない。
学習のvalidation lossは開始0.133、step128で0.139であり、loss低下を採用根拠にしていない。

## 新規16問の最終比較

候補を固定してから生成した新規16問は、従来の親14/16、新system＋補助情報のみの親11/16、
学習したstep64が16/16だった。strict JSONはいずれも16/16。
候補では金額・時間上限、人数下限、比較境界、未確認・確認済み状態を保持した。
この小規模な集合での改善を、任意の文章や自由討論の正確さの証明とは扱わない。
旧回帰・実討論は以下のとおり完了したが、既定への昇格はしていない。

候補の同じ本文1問で、Adapter着脱前後のBaseのrawが一致した。
これは1問の本文probeに限る確認であり、全ての判断・投票の非回帰は別に確認する。

## 旧回帰99例

原文・正解・アンカーを変えず、従来の親と候補を同じ既存集合で生成比較した。
保存raw全198件を共通採点器で再検査し、全チェック・集計が一致した。

| 集合 | 従来の親 | v6 step64 |
|---|---:|---:|
| v5の16問 | 15/16 | 14/16 |
| 確認済み効果 | 8/8 | 7/8 |
| 行為と効果 | 14/16 | 9/16 |
| 時制 | 14/16 | 15/16 |
| 利用条件 | 13/16 | 12/16 |
| 初期12問 | 11/12 | 10/12 |
| email EV bike既存 | 15/15 | 15/15 |

不合格には「試めています」「試んでいます」等の不自然な活用、常体語尾、固定表現への不一致があった。
全てを事実反転と扱うわけではないが、自然な本文の非回帰として受け入れない。
初期12問の語尾補正込みは11/12で、コード補正の1件を直接のWeight成功へ加えていない。

既存15問のcold-depot文には、元の「利用不能が集中する寒冷depotでpilotを先に行う」から
「寒冷depotでpilotを先に行い、利用不能が集中します」となった例もあった。
これは粗い検査では通るが、問題の所在を述べる修飾節が別の主述関係へ変わっており、
合格数だけで意味保持を認定できない。元rawは保持し、根拠なく点数を書き換えていない。

## 8人の実討論

同じ架空のSwift/Rust台帳、判断temperature 0、本文temperature 0、seed 20261005、
最大24 event・2 round、話者別cacheで親・候補を各1回実走した。
独立8人、15 eventの対案交換、1 roundのすり合わせまで完走した。

| 実測 | 従来の親 | v6 step64と補助情報 |
|---|---:|---:|
| 実行時間 | 238.105秒 | 211.900秒 |
| model calls | 33 | 33 |
| Weight本文 | 19/23 | 23/23 |
| 凍結claimからの代替 | 4 | 0 |
| 失格claim | 1 | 1 |
| whole discussion hard gate | fail | fail |

候補では「備える」の提案を「備えます」と生成し、完了形・進行中への変更を避けられた。
全15本文入力のkindは台帳由来のproposalで、数量補助情報も原文由来であることを照合した。
語尾補正・schema修復は0件。同意・対案の導入句はコード合成で、本文部分だけがWeight生成である。
単発の実行時間は他処理・cache・生成内容の影響もあり、Weightによる速度改善の証明ではない。

初期8人のraw、投票8人のraw、最終選択は親と候補で一致した。
これは同じ1組のfixtureの確認であり、任意の将来判断への非回帰証明ではない。
本文の補助情報はBaseの判断promptへ入れない。
BaseのD05を移植不要、D06を安全性確保と扱う誤解は引き続き残り、
RUST_FIRSTの未許可D05は両runで拒否した。ID検査の成功を、全説明の事実確認とは扱わない。

今回の改善は、補助情報と学習を組み合わせた本文contractの一部である。
新規と実走の成功を根拠に旧集合の後退を無視せず、既定置換・Weight公開を行わない。

## 保存先

```text
dataset: /Volumes/data4/cod_model_weight/datasets/claim-body-qwen35-4b-v6/mlx_shared
training: /Volumes/data4/cod_model_weight/adapters/claim-body-qwen35-4b-v6/hints
audit: /Volumes/data4/cod_model_weight/evaluations/claim-body-qwen35-4b-v6_20261005
```

`evaluation_plan.json`と`source_snapshot`に、学習前の3条件、選定規則、入力契約とsourceのSHAを保存した。
全95単体テストが成功したが、学習完走・Weight性能の向上・実討論の確認とは区別する。
[教材](../data/general_body_qwen35_v6/curated.json)、[system](../configs/claim-body-qwen35-4b-v6-system.txt)、
[学習設定](../configs/claim-body-qwen35-4b-v6.yaml)を参照できる。
[非昇格記録](../promotions/qwen3.5-4b-claim-body-v6-step64.json)にはSHA・条件・未達理由を保存した。
