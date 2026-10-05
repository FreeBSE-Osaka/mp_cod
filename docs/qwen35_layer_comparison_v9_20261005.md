# Qwen3.5 v9 学習対象層の比較

## 検証状況

入力診断で、追加system文が既知の包含反転を誘発した例があった。
今回の比較は元の本文systemを通常・補助入力で共通にし、MLPだけの学習とattentionも追加する学習を分離する。
2026年10月5日から6日にかけて実モデルの初期化・入力検査を完了し、MLP条件は48ステップを327.815秒で正常完走した。
MLPのpeak memoryは6.442GB、attention追加も48ステップを371.583秒で正常完走し、peakは7.687GBだった。
両条件の学習token数は1,764で一致した。
開発比較は完了し、事前規則でMLP48を最終生成前に選定・固定した。
新規・全旧・診断例の比較と、通常／補助の親・候補8人実討論4条件を完了した。
新規精度改善は確認できず、個別後退とBaseの根拠誤読が残る。研究HOLD、既定置換・Weight公開なし。

## 固定条件と比較条件

BaseはQwen3.5-4B-4bit、両方の親はv8 step144の同じMLP Weight。
同じtrain2,560表現、原文・話者組1,280、元system、seed20261005、batch1、rank4、scale8、
learning rate5e-6、最後4層、48ステップ、gradient checkpoint有効を使用する。
通常・補助入力は同じ正解であり、systemだけも同じ。
片方はmlp.down_projのみ、もう片方は同じMLPと次の入力投影を学習する。

- linear attentionのin_proj_qkv
- 通常attentionのq_projとv_proj

Qwen3.5の最後4層は、linear attention3層と通常attention1層であることを実装と実モデルで確認した。
Native MLX-LMの同じ学習器とNumPy seedによる同じバッチ順序を用い、2条件は逐次実行する。
新しい汎用trainerやモデル振分けを追加しない。
対象層の追加でパラメータ数も増えるため、特定のattention投影だけの独立効果とまでは断定しない。

## 親Weightの保持と初期化

新しい構成の前に、親MLPの8テンソルが一致し、追加する10テンソルのB行列が全てゼロであることを確認した。
追加5モジュールは最後の3 linear attentionと1通常attentionのq/vである。
同じ決定論的本文1例の生成は追加前後で完全一致した。
これは1例の出力probeと親テンソルの全保持の確認で、全入力の非回帰を証明するものではない。

親MLP Weightは754,632bytes、追加初期Weightは1,501,361bytesだった。
初期WeightのSHA256は`f9b5fc70be5247fd36d711edf055e6e2a99828e4ce0cd790fae90ea7fa67ce6a`。
単独実行できるモデルではなく、同じBaseへ適用する本文用LoRAである。

## データ分離と実入力検査

学習はv8のtrain96件だけと、以前のv3 train512組の復習で、以前の開発・最終・入力診断の答えを学習へ戻していない。
硝子・毛筆の新規dev8と、製菓・木工・帆布・竹細工のtest16を別topicで固定した。
両splitに8話者を含み、過去corpusと完全一致しない。
新しい正解24件は全件合格し、反転・過去化・試行意図欠落等の誤文24件は全件不合格だった。

実MLXのChatDatasetで全4,140表現の正解、思考なしの入力prefix、JSONだけの学習対象、
同じ正解とsystemの入力対、split分離を確認した。
最大tokenはtrain293・valid285・test292で、448以内だった。
親v8の新規dev8は通常・補助とも7/8で、過去形節の丁寧語表現が不一致の1件が残る。

## 選定と必須検証

両条件の24・48ステップを同じdev8・二入力で比較する。
重大違反が少ない候補、合計16出力の直接合格が多い候補、同点なら少ないtrainable tensor、早いstepの順に選ぶ。
重大違反はschema・構文・中立性・整合・数値・固定アンカー・競合選択の検査で、全不一致を意味反転とは呼ばない。
選定を最終生成前に保存し、testを見て別Weightへ変更しない。

新規16問、全旧147例、既に使った入力診断例、個別の親合格保持、Adapter着脱probe、
8人の実討論を同じ選定Adapterの両入力で確認する。原文・正解・アンカーを変更しない。
修復・代替文をWeightの直接成功に数えず、失敗条件を別モデルへ振り分けない。
BaseのD05/D06源データ誤解と自由な自然討論の未達も残しており、この小規模比較だけでGoalを完了にしない。

## 再現と保存先

開発8問の通常／補助入力は、MLP24が6/8・7/8、MLP48が7/8・7/8、attention24が6/8・7/8、
attention48が7/8・7/8だった。対応する32出力のrawはMLPとattentionで全て一致した。
5つの追加attentionモジュールのB行列は全て非ゼロに更新されており、未学習の設定ではない。
ただし48ステップとこの集合では、追加層の生成品質改善は確認できなかった。
全入力で同等、attentionが常に無効、といった一般結論にはしない。

同点時の少ないtrainable tensor・早いstepの事前順序でMLP48を選んだ。
親の開発成績とも同点であり、選定を昇格や性能改善の証明とは扱わない。
MLPとattentionのvalidation lossは開始0.059、24で0.156／0.157、48で両方0.157だった。

新規16問の親対照は通常・補助とも13/16、選定したMLP48も両入力13/16だった。
追加学習の精度改善はこの集合で確認できなかった。
過去の試行を「試みて」にして時制を落とす例、計測項目を省く例が残り、共通guard・固定アンカーで拒否した。
旧147例は同じ元systemで親v8が通常141/147・補助139/147、候補が通常139/147・補助140/147だった。
通常の親合格5例、補助の親合格2例を失っており、集計の増加を非回帰成功とは扱わない。
通常の1例は「全mailへ展開します」を「展開しました」に変える意味の後退だった。
ほかに試しています／試みています、保存しません／保存せず等の固定アンカー不一致があり、
これらを全て意味反転と呼ばず、原文とrawを別々に確認する。
通常の対照は入力・話者・system・Weight SHAが一致した保存rawを再利用し、現行guardで別ファイルへ再採点した。
補助の親対照147件は元systemで新しく生成した。以前の追加systemでの成績とは混同しない。
既知の売上除外例は親・候補の両入力で保持した。
元v3の旧131件の保存rawと、不足していた旧v8 test16件の新規生成を照合し、
同じ元systemの全147対照を完成させた。元v3は130/147、v7の通常対照は137/147だった。
候補は元v3の通常合格3例・補助への比較で1例を失っている。補助対通常の比較は入力が違う参考値で、
同じ補助入力の親v8との比較と区別する。
保存された評価22ファイル・1,173行（再利用・同じrawを含む集計）で、
原文・話者・正解を含まない完全なrequestと、現行guardによる採点が一致した。
これは1,173回の新規生成という意味ではない。

以前に使用済みの診断12件は、同じ元systemで親が通常・補助とも7/12、候補が両方9/12だった。
両入力で親の合格を保持したが、新しいholdoutではない。過去形の欠落と計測項目の省略3件が残る。
通常・補助の親／候補8人実討論4条件も完了した。いずれも本文契約は通ったが、全体は不合格だった。

## 数値に出ない意味の後退

同じ13/16でも、過去試行の1例は親の「仕上げ台の変更を試みたが、崩れへの効果は未検証です。」から、
候補の「仕上げ台の変更を試みて、崩れへの効果は未検証です。」へ変わった。
親も固定アンカーは不一致だったが、過去形は保持していた。候補は明示的な過去の状態を失った。
「どちらも失敗」とだけ集計すると、この悪化が見えない。

製菓の計測項目「仕上げ時間37分」は「箱代も含めて37分で仕上げます」に変わる例が残る。
数字が一致していても、計測対象を作業完了の主張へ変えると同じ意味ではない。
固定アンカーは拒否したが、一般の数値一致guardだけで全ての意味を保証できるわけではない。

## 自然な議論との境界

本文LoRAは、検証済みのclaimを一文にするrendererであり、議論の理由・根拠選択・投票を学習したWeightではない。
「その案に賛成です」「別の進め方として」等の接続句は`compose_dialogue_body`が付ける。
`model_body_v2`というoriginは本文の出自を示し、発言の全てが自由生成されたことを意味しない。
23発言がWeight由来でも、実生成15件とキャッシュ再利用8件を分けて記録する。

今回の固定fixtureはSwift／Rustの2案で、同じ案の言い換えが多い。
その本文契約が通っても、多様な理由・条件付き賛成・否定のみ・新しい対案が自然に交換されるGeneral討論の合格ではない。
機械的発言の簡易指標が0でも、この質的な制約は残る。

## 8人の実討論4条件

同じ固定ledger、8人格、seed20261005、temperature0、話者別cache、24event上限、2回すり合わせ上限で実行した。
実際には15eventと8人の投票後、全員がSwift案に合意したため1回ですり合わせを終了した。
Adapterは本文だけに適用し、独立判断・根拠・投票のAdapterは全てNoneだった。

| 条件 | 時間 | model call | 直接raw本文 | Weight由来発言 | 代替文 | 全体gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 親v8 通常 | 342.036秒 | 33 | 15/15 | 23/23 | 0 | 不合格 |
| 候補v9 通常 | 280.608秒 | 33 | 15/15 | 23/23 | 0 | 不合格 |
| 親v8 補助 | 310.668秒 | 33 | 15/15 | 23/23 | 0 | 不合格 |
| 候補v9 補助 | 351.371秒 | 33 | 15/15 | 23/23 | 0 | 不合格 |

本文のschema修復・語尾補正は全条件0。23発言は15の直接生成と8のcache再利用で、23回生成したという意味ではない。
各入力の親／候補で独立判断raw8/8、投票raw8/8、投票修復raw、最終選択が一致した。
これは同じ1つの決定論的fixtureでの分離確認であり、将来の全入力のAdapter分離を証明するものではない。
実行順と他プロセスの影響を分離しておらず、時間差をWeightによる高速化とは解釈しない。

全条件でRUST_FIRSTへの不適合D05が1件拒否された。
さらに、D05は移植不要とはしていないのに、Baseが「D05 の移植不要前提」と生成する誤読が残った。
D06は速度・安全性の優劣が未実測なのに「D06 の安全性確保」とする根拠付けも残る。
IDの存在・許可だけで、根拠の意味を正しく読めたことにはならない。本文LoRAはこの判断を変更しない。

候補の実生成本文の一例は「最初の版はSwiftのコアで早く仕上げ、将来の移植は追加課題として残します。」だった。
表示される「その案に賛成です。」はコード側の接続句である。
完全な発言・根拠・rawはaudit配下の`language_parent_plain`、`language_selected_plain`、
`language_parent_hints`、`language_selected_hints`の`event_debate_*.json`に保存した。

## 今回の判断

MLPとattentionの実学習、初期化・更新、全回帰、個別非回帰、Adapter着脱、8人実走を確認できた。
しかし新規集合の精度改善も自由な議論の改善も確認できず、候補は昇格しない。
次に扱う課題は、数字と計測項目の結び付き、文中の過去試行、Baseの否定・未確定情報の誤読、
コードの接続句とclaimの言い換えだけに留まらない自然な理由の交換である。
今回の4条件の完走を、General全体の完成とは扱わない。

dataset作成は既存buildの両systemに同じ元systemファイルを指定する。
MLP条件は親v8のWeight、attention条件はゼロ追加を検証した初期Weightから再開する。

```sh
<mlx-python> tools/general_body_training.py build \
  --curated data/general_body_qwen35_v9/curated.json \
  --renderer-system-file configs/claim-body-qwen35-4b-v5-system.txt \
  --plain-system-file configs/claim-body-qwen35-4b-v5-system.txt \
  --constraint-hints --rehearsal /path/to/v3/mlx_shared --out /path/to/new-dataset

<mlx-python> tools/general_body_training.py train \
  --model /path/to/Qwen3.5-4B-4bit --data /path/to/new-dataset \
  --config configs/claim-body-qwen35-4b-v9-mlp.yaml \
  --parent-adapter /path/to/v8/step144/adapters.safetensors --out /path/to/new-mlp

<mlx-python> tools/general_body_training.py train \
  --model /path/to/Qwen3.5-4B-4bit --data /path/to/new-dataset \
  --config configs/claim-body-qwen35-4b-v9-attention.yaml \
  --parent-adapter /path/to/bootstrap_attention/adapters.safetensors --out /path/to/new-attention
```

既存Python3.11 MLX環境、外付けTMPDIR、オフライン指定、20GiB・2秒の容量監視を維持する。
内蔵へBaseを複製せず、別Model推論と並行学習をしない。

```text
dataset: /Volumes/data4/cod_model_weight/datasets/claim-body-qwen35-4b-v9/mlx_dual_fixed_system
adapters: /Volumes/data4/cod_model_weight/adapters/claim-body-qwen35-4b-v9
audit: /Volumes/data4/cod_model_weight/evaluations/claim-body-qwen35-4b-v9_20261005
```

`evaluation_plan.json`、`bootstrap_probe.json`、`token_audit.json`、`source_snapshot`に生成・学習前の条件を保存した。
`evaluation_audit.json`、`diagnostic_comparison.json`、`manual_semantic_review.json`、
`source_statement_review.json`、`runtime_comparison.json`に最終照合を保存した。
重みとローカル監査スクリプトはGitへ含めず、公開は教材・設定・テスト・記録のみ。
attentionの初期化スクリプトはaudit配下の`bootstrap_attention.py`にあり、公開済み単独Weightパッケージではない。
全101単体テストは成功したが、Weightの改善の確認とは区別する。
[新規問題](../data/general_body_qwen35_v9/additions.json)、[全教材](../data/general_body_qwen35_v9/curated.json)、
[MLP設定](../configs/claim-body-qwen35-4b-v9-mlp.yaml)、[追加attention設定](../configs/claim-body-qwen35-4b-v9-attention.yaml)を参照できる。
[選定WeightのSHAと非昇格理由](../promotions/qwen3.5-4b-claim-body-v9-mlp-step48.json)も保存した。
