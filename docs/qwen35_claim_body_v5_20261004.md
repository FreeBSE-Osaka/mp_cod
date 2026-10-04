# Qwen3.5-4B Claim Body v5 学習と検証

## 検証状況

General CoDの本文を生成するLoRAの追加学習を2026-10-04に開始した。
128ステップを2026-10-05に正常完走した。実行時間は619.145秒、MLX報告peak memoryは6.034GB。
開発用でstep32を選定し、最終検証前に固定した。新規16問は親・候補とも15/16。
候補には予算上限の欠落があり、得点が同じでも昇格条件は未達。
旧集合の比較は完了し、以前の時制16問は14/16→16/16へ改善した。
一方、初期12問は11/12→10/12となり、こちらも時間上限を落とした。
8人討論も完走したが、Weight本文の直接採用は19/23→17/23へ悪化した。
v5は不採用の研究Weightとして保持し、親を置き換えない。
研究用HOLDであり、既定Weightの置換やWeight公開はしていない。
この文書はプロンプトだけのPDCAとは区別して、実際の学習と検証を記録する。

親はv3 step64。v4の長いsystemは非回帰が悪かったため採用せず、
親と同じ288文字のsystemで追加学習する。Baseは主張・根拠・投票を決め、
LoRAは検証済みclaimの本文だけを生成する分離を維持する。
今回も一つの共有Adapterで8話者を扱い、8個のBaseモデルを同時にロードする構成ではない。

## 教材と学習設定

新規の架空教材は厨房、修理、閲覧室、講習の32件で、8話者へ展開した256例。
v3のtrain全512例を復習に使い、学習は768例とした。
計画・進行中・完了、効果の未確認・確認済み、限定条件、割合と金額の単位を区別する。
過去のモデルの誤出力や反省文を正解として学習へ戻していない。

新しい開発用8件と最終検証16件は別topicで、両方に8話者を含む。
以前の時制・条件・効果検証・PDCAのclaimとの完全一致はない。
学習・開発・最終検証のclaimも重複しない。
事前に書いた正解と反例を検査し、反例はSFTの正解に含めない。
数字の空白等を許す新規アンカーは生成前に固定し、既存集合のアンカーは変更しない。

設定は最後4層の`mlp.down_proj`、rank 4、scale 8、batch 1、learning rate 1e-5、
128ステップ、gradient checkpoint有効。親から引き継ぐのはAdapter Weightで、
optimizer状態を再開するものではない。
最大入力はtrain 259、valid 265、test 266 tokensで、設定448以内だった。

開発用での親の直接合格は6/8、strict JSONは8/8。
「試している」の進行状態を明示しない文と、正しい別活用が固定表現へ一致しない文が残った。
固定表現の失敗だけを意味反転と扱わず、実際のrawも確認する。

step32・64は開発用7/8となり、「試しています」の進行状態を保つようになった。
残る1件は「再見積ります」と「再見積りします」の活用表現の差によるアンカー不一致。
step96・128は6/8だった。重大な検査違反は全stepで0件で、同点なら早いstepという事前ルールで32を選んだ。
最終検証はまだ終わっておらず、この開発結果だけで採用しない。
学習のvalidation lossは開始0.110、step128で0.112であり、loss低下を成果の根拠にしていない。

## 学習と推論の入力境界

MLX-LM 0.31.3の学習datasetは、既定の思考ありpromptを使う。
本文推論は`enable_thinking=False`であり、思考タグを閉じてからJSONを生成する。
今回確認した例では全token列は同じでも、従来の学習対象には思考タグの閉じ部分が2 tokens含まれた。
学習対象の開始位置を推論の開始位置と揃える修正を行った。
これが従来の意味誤生成の原因だったと断定するものではない。

最初の調整は、MLXラッパーへの代入が内側へ転送され、外側のmethodを再び呼ぶ再帰エラーになった。
学習step1前に終了し、Weightは生成されなかった。
二回目は内側へ設定したが、MLXが明示的に渡す`True`が既定値の`False`を上書きした。
40ステップの報告後に中断し、保存済みstep32は選定から除外した。
失敗・中断のログとsourceは保持し、正常完走と扱わない。

修正版は内側の呼出しで明示的に`False`を優先する。
実際のMLX TokenizerWrapperで全1,558例を検査し、学習promptのtoken境界と推論promptが一致し、
学習対象がbodies JSONから始まることを確認した。外側から`True`が渡っても変更されない。
修正版は除外試験のWeightを引き継がず、親v3から別ディレクトリへ学習する。

## 選定と昇格条件

32・64・96・128のcheckpointを新規dev8で生成比較する。
schema・構文・丁寧語・意味整合・数値単位・競合主張の重大違反が少ない候補を優先し、
次に直接合格数、同点なら早いstepで選ぶ。lossだけでは選ばない。
候補を固定してから最終検証を生成し、その結果を見て別stepへ差し替えない。

最終検証には新規16問、確認済み効果8問、以前の時制・条件・行為効果、初期12問・既存15問の非回帰、
Adapter着脱前後のBase確認と8人の話者別討論を含む。
語尾補正・凍結claimからの代替文は、Weightによる直接生成成功に数えない。
生の生成文と根拠の意味も確認し、小規模な文字列検査の合格を任意の自由討論の正確さとは扱わない。

## 新規16問と数量条件のガード

同一Base・system・temperature 0で、親と候補はどちらも直接15/16、strict JSONは16/16。
候補は「録音位置の変更を試している」の進行状態を明示するようになったが、
別の文で「共有用具の予算は2600円まで」を「共有用具の予算は2600円」と生成し、上限条件を落とした。
固定アンカーは拒否したものの、実運用で使う`body_matches_claim`は当初この文を通していた。
集計の同点を非回帰や安全性の証明にしない。

共通の本文検査へ、明示的な数量条件の保存を追加した。
「まで・以下・上限」等の同等表現を許し、上限の欠落、下限への反転、
「超える」と「以上」等の境界変更を拒否する。
数値はFractionで比較し、割合表記、人数・週の単位別名、「上限は1回18人」の複合表現も検査する。
一般的な主語と数値の対応や任意の否定・条件文全体を理解する検査ではなく、意味レビューは引き続き必要。

元のrawと採点は保持し、親・候補へ同じ強化guardを適用した再採点を別ファイルへ保存した。
新規16問の直接合格数は両方とも15/16のまま。コードで失敗を直してWeightの合格数を増やしていない。
上限欠落は実運用の本文整合でも拒否され、凍結claim由来の代替文へ戻る。
代替文をWeight生成の成功とは数えない。
既存の手作成正解とSFT全1,558例が強化guardで通ること、全92単体テストも確認した。

同じ本文1問のBase出力はAdapter着脱前後で完全一致した。
これは1問の本文probeで、全ての判断・討論の非回帰を証明するものではない。

## 以前の集合との比較

元のtopic・claim・正解・アンカーを変えずに回帰集合をまとめ、Baseの再ロードを減らして各83例を生成した。
結果は集合別に確認し、全体の合格率へまとめて一般能力とは扱わない。
同じ強化guardで親と候補の保存raw全166件を再採点し、元のチェックと一致した。

| 集合 | 親v3 | v5 step32 |
|---|---:|---:|
| 確認済み効果 | 8/8 | 8/8 |
| 以前の行為と効果 | 14/16 | 15/16 |
| 以前の時制 | 14/16 | 16/16 |
| 以前の利用条件 | 13/16 | 13/16 |
| 初期12問 | 11/12 | 10/12 |
| email EV bike既存 | 15/15 | 15/15 |

以前の「散策経路を試している→検証済み」という確認状態の反転は候補で出なかった。
一方、初期12問の「1回25分まで」を「1回25分」とした文は、数量条件guardが拒否した。
新規16問の予算上限と別topicでも起きており、上限条件の保持を次の課題とする。
最終検証の失敗を見て別checkpointへ差し替えていない。

## 8人討論の実走

同じ架空のSwift/Rust選定を、親と候補で各1回実走した。
8人全員が両案を選べ、独立初期主張、15 eventの対案交換、1 roundのすり合わせ投票まで進んだ。
判断temperature 0、本文temperature 0、seed 20261004、最大24 event・2 round、話者別cache。
反対だけの役へ固定したり、人数の同意を独立した証拠と数えたりしていない。

| 実測 | 親v3 | v5 step32 |
|---|---:|---:|
| 実行時間 | 179.374秒 | 181.407秒 |
| model calls | 33 | 33 |
| Weight本文 | 19/23 | 17/23 |
| 凍結claim由来の代替文 | 4 | 6 |
| 失格claim | 1 | 1 |
| whole discussion hard gate | fail | fail |

初期8人のraw・投票8人のraw・最終選択は両runで一致した。
判断callのAdapterは全てNoneで、候補Adapterは本文にだけ使用した。
これは同一条件の1組の討論における確認であり、将来の全ての判断への非回帰証明ではない。
以前のtemperature記録の異なるrunを、この比較の親として流用していない。

候補の6件の代替は、提案の「将来の移植に備える」を「備えました」という完了済みの事実へ変えたもの。
新規・過去の本文テストでの改善が、話者別の実討論へ十分に移っていない。
同意や対案の接続部分はコードが合成し、本文部分がWeightの生成である。
同じclaimを複数話者で出しても文体は非常に似ており、自由な自然討論の完成とは扱わない。

BaseのrawにはD05を「移植不要」、D06を「安全性確保」と扱う誤りも残った。
実データは移植工数が未算定・速度や安全性の優劣が未実測であり、D番号が正しくても説明の意味は保証されない。
RUST_FIRSTの未許可D05は両runで拒否したが、この拒否成功だけで全理由の裏付け確認済みとはしない。
短い自然文・内部protocol・数量guardも、任意の主張の正しさの証明にはならない。

Baseと親AdapterのファイルSHAは作業前後で不変だった。
比較の原文は`runtime_parent.log`、`runtime_selected.log`と各run JSONに保存し、
`runtime_comparison.json`へ条件一致と限定した比較結果を記録した。

## 容量監視と再現

`tools/general_body_training.py`のCLIは、作業先・出力先・一時領域を20GiB・2秒間隔で監視する。
容量不足は終了コード2、通常中断は130として、出力先に対応する`*.job.json`へ記録する。
既存journal・非空の学習出力先を上書きせず、バックグラウンドthreadからの容量監視実行も拒否する。
OS全体のメモリ圧迫を保証する機能ではなく、他アプリやOllamaサーバー全体を終了しない。

元の外付けBaseモデルを直接ロードし、内蔵へ数GBのコピーは作らない。
一時領域も外付けへ設定した。今回の学習開始時、内蔵約28GiB・外付け約30GiBを確認した。

```sh
TMPDIR=/Volumes/data4/cod_model_weight/tmp \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
<mlx-python> tools/general_body_training.py build \
  --curated data/general_body_qwen35_v5/curated.json \
  --renderer-system-file configs/claim-body-qwen35-4b-v5-system.txt \
  --rehearsal /path/to/v3/mlx_shared --out /path/to/new-dataset

TMPDIR=/Volumes/data4/cod_model_weight/tmp \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
<mlx-python> tools/general_body_training.py train \
  --model /path/to/Qwen3.5-4B-4bit --data /path/to/new-dataset \
  --config configs/claim-body-qwen35-4b-v5.yaml \
  --parent-adapter /path/to/v3/step64/adapters.safetensors \
  --out /path/to/new-adapter

<mlx-python> tools/general_body_training.py evaluate \
  --model /path/to/Qwen3.5-4B-4bit --adapter /path/to/checkpoint \
  --renderer-system-file configs/claim-body-qwen35-4b-v5-system.txt \
  --curated data/general_body_qwen35_v5/curated.json --split valid \
  --out /path/to/new-dev.json
```

学習には本リポジトリの`train`を使う。MLX-LMのCLIを直接呼ぶだけでは、
今回の入力境界修正と容量journalを適用しない。
評価は引き続き思考なし、temperature 0、最大160 tokens。
最後のWeightと各stepのファイルを混同せず、選定stepとAdapter configを同じディレクトリへ保存する。

## 保存先

```text
dataset: /Volumes/data4/cod_model_weight/datasets/claim-body-qwen35-4b-v5/mlx_shared
training: /Volumes/data4/cod_model_weight/adapters/claim-body-qwen35-4b-v5/aligned_r3
audit: /Volumes/data4/cod_model_weight/evaluations/claim-body-qwen35-4b-v5_20261004
```

`evaluation_plan.json`、`tokenizer_alignment_audit_r3.json`に選定規則、source・親・Base・教材のSHA、
除外試験の理由と実Tokenizer検査を保存した。各試行のsourceも別に保持している。
固定の教材・アンカー・Baseの判断ロジックは変更していない。
本文の数量条件guardは上記のとおり強化し、変更前後のsourceと採点を区別して保存した。
92単体テストの成功は、Weightの一般性能向上や8人討論の成功を証明するものではない。

[教材](../data/general_body_qwen35_v5/curated.json)、[学習設定](../configs/claim-body-qwen35-4b-v5.yaml)、
[固定system](../configs/claim-body-qwen35-4b-v5-system.txt)を参照できる。

[非昇格記録](../promotions/qwen3.5-4b-claim-body-v5-step32.json)にWeight SHA、選定、各検証、未達理由をまとめた。
次は、入力の提案・事実の区別と数量条件を明示する方式を開発用で切り分け、
別の教材・未使用問題で学習し、全非回帰と8人討論を再検証する。
今回の最終検証の答えや誤ったBaseの理由文を、そのまま次の正解として学習しない。
