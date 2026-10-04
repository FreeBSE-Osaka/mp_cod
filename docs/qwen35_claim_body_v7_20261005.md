# Qwen3.5-4B Claim Body v7 両入力復習と非回帰検証

## 検証状況

v6で改善した新規文と実討論を維持し、旧表現の後退を直すため、通常入力と補助入力を同じAdapterへ復習させる。
256ステップを1366.919秒で正常完走し、MLX報告peak memoryは7.554GBだった。
開発用の二入力でstep256を選び、新規16問・旧115例・8人実討論まで完了した。
通常入力の旧115例は親104/115から110/115へ改善し、親の合格例も全て保持した。
補助入力は107/115だが、除外条件の反転を含む4例の個別後退が残った。
実討論の本文は19/23から23/23へ改善したものの、Baseの根拠誤読が残り、討論全体は不合格。
研究用HOLDで、既定モデル・prompt・Weightを置き換えず、Weightを公開しない。

実行時に旧モデルへ振り分けるルーターは作らない。
一つの選定Adapterを両入力と全回帰で検証し、失敗した集合を除いて成功と扱わない。
Baseの主張・根拠・投票は引き続き本文Adapterから分離する。

## Promptと事前比較

通常入力では従来の288文字systemを使用する。
補助入力のsystemは、その288文字を完全なprefixとして保持し、
kind=proposalと数量比較を保つ短い注意だけを追記する。
v6のような全面的なsystem書換えを避けることが、旧表現を維持するかを検証する。

v6 dev8を使った学習前診断では、旧systemを残す補助入力は親で7/8、
v6の書換えsystem＋補助入力は5/8だった。
これは既に使った開発問題での原因切り分けで、新しい盲検試験の成績とは呼ばない。

別に用意したv7 dev8の親baselineは、通常入力5/8・補助入力6/8。
上限欠落、常体節、固定表現の違いが残った。
同じ新しい問題を各checkpointの通常・補助入力で生成し、合計16出力で選定する。

## 教材と両入力の組

v6の学習用4 topic・32件だけを復習し、別の備蓄・付箋の活用と確認状態16件を追加した。
「試みる」「試みている」「確かめる」「確かめた」等を区別する。
v6のdev/testの答え、モデルの誤った生成文や反省文を正解へ取り込まない。
新しいv7 dev8・test16は別topicで、両方に8話者を含み、以前のclaimと完全一致しない。

48件を8話者へ展開した384例と、v3のtrain復習512例で896の原文・話者組を作る。
各組を通常入力・補助入力の2形へ展開するが、原文・話者・正解JSONは同一である。
学習は1,792表現、validは566表現、testは1,014表現。
二形を独立した1,792の新規問題とは数えない。

kindと数量比較の補助情報は既存の原文・明示kindから作り、kind不明は未指定のままにする。
正解文やアンカーを補助情報へ入れない。
通常入力には補助情報を付けないので、元の入力契約も同じWeightへ復習する。
両形の出力契約は従来のbodies JSONのままである。

実MLXで全3,372表現について、同じ正解の対、入力境界、JSON本文だけの学習対象、正解の検査、
train/valid/testのclaim分離を確認した。
最大tokenはtrain330・valid325・test329で、設定448以内。

## 学習と選定規則

親はv3 step64。最後4層のmlp.down_proj、rank4、scale8、batch1、
learning rate5e-6、256ステップ、gradient checkpoint有効、seed20261005。
学習率をv6の半分に下げた効果と両入力復習を含む実験で、各変更の独立した因果効果までは断定しない。

64・128・192・256の各Weightを、同じdev8・二入力で生成する。
合計の重大違反が少ない候補、直接合格数が多い候補、同点なら早いstepという事前順序で選ぶ。
lossだけで選ばず、最終生成前に保存して固定する。
testや旧回帰の結果を見て別stepへ変更しない。

## 再現

```sh
<mlx-python> tools/general_body_training.py build \
  --curated data/general_body_qwen35_v7/curated.json \
  --renderer-system-file configs/claim-body-qwen35-4b-v7-system.txt \
  --plain-system-file configs/claim-body-qwen35-4b-v5-system.txt \
  --constraint-hints --rehearsal /path/to/v3/mlx_shared \
  --out /path/to/new-dual-dataset

<mlx-python> tools/general_body_training.py train \
  --model /path/to/Qwen3.5-4B-4bit --data /path/to/new-dual-dataset \
  --config configs/claim-body-qwen35-4b-v7.yaml \
  --parent-adapter /path/to/v3/step64/adapters.safetensors \
  --out /path/to/new-adapter

<mlx-python> tools/general_body_training.py evaluate \
  --model /path/to/Qwen3.5-4B-4bit --adapter /path/to/checkpoint \
  --renderer-system-file configs/claim-body-qwen35-4b-v5-system.txt \
  --curated data/general_body_qwen35_v7/curated.json --split valid \
  --out /path/to/new-plain-dev.json

<mlx-python> tools/general_body_training.py evaluate \
  --model /path/to/Qwen3.5-4B-4bit --adapter /path/to/same-checkpoint \
  --renderer-system-file configs/claim-body-qwen35-4b-v7-system.txt \
  --curated data/general_body_qwen35_v7/curated.json --split valid \
  --constraint-hints --out /path/to/new-hints-dev.json
```

`--plain-system-file`はdataset作成専用で、実行時の自動モデル選択ではない。
指定時は`--constraint-hints`と空でないsystemを要求する。
指定しない既存build動作は維持する。
出力とjournalの上書きを拒否し、CLIは20GiB・2秒間隔の容量監視を使う。
今回もモデルを内蔵へコピーせず、一時領域と大きい成果物は外付けへ置く。

## 開発用の選定結果

| Weight | 通常入力 | 補助入力 | 合計 |
|---|---:|---:|---:|
| 親v3 step64 | 5/8 | 6/8 | 11/16 |
| v7 step64 | 5/8 | 6/8 | 11/16 |
| v7 step128 | 5/8 | 6/8 | 11/16 |
| v7 step192 | 5/8 | 6/8 | 11/16 |
| v7 step256 | 6/8 | 6/8 | 12/16 |

事前順序でstep256を選び、最終生成前に`selection.json`へ固定した。
ここでの重大違反はschema・構文・中立性・整合・数値・固定アンカー・競合選択の検査であり、
全てのアンカー不一致を意味反転と呼ぶものではない。
validation lossは開始0.113、step64で0.104、step256で0.137。
lossが最小のstepへ、最終結果を見て差し替えていない。

## 新規16問の最終比較

| 条件 | 直接合格 | strict JSON |
|---|---:|---:|
| 親と通常入力 | 15/16 | 16/16 |
| 親と補助入力 | 13/16 | 16/16 |
| 同じv7 step256と通常入力 | 15/16 | 16/16 |
| 同じv7 step256と補助入力 | 15/16 | 16/16 |

通常入力は親と同点、補助入力では同じ入力の親より2件改善した。
残る1件は「配色変更を試みているが」の文中常体と固定アンカーの不一致で、
全問成功とは扱わない。語尾補正を加点していない。
Adapterを取り外した前後でBaseのrawが一致したが、これは1問の決定論的本文probeだけの確認である。

## 旧115例と個別の非回帰

原文・正解・固定アンカーを変更せず、同じ選定Adapterを両入力で検査した。
親側はv6の通常入力評価99例と新規16例の保存rawを再利用した。
原文、話者、完全な入力、WeightのSHA、現在の採点結果が一致することを全115例で照合しており、
再生成した親baselineとは記載しない。

| 集合 | 従来の親 | v7通常入力 | v7補助入力 |
|---|---:|---:|---:|
| v6の16問 | 14/16 | 16/16 | 16/16 |
| v5の16問 | 15/16 | 16/16 | 15/16 |
| 確認済み効果 | 8/8 | 8/8 | 8/8 |
| 行為と効果 | 14/16 | 15/16 | 14/16 |
| 時制 | 14/16 | 16/16 | 14/16 |
| 利用条件 | 13/16 | 13/16 | 14/16 |
| 初期12問 | 11/12 | 11/12 | 11/12 |
| email EV bike既存 | 15/15 | 15/15 | 15/15 |
| 合計 | 104/115 | 110/115 | 107/115 |

各集合の合格数は親以上だが、集合内で失敗が入れ替わる可能性も別に確認した。
通常入力は親の合格104例を全て保持し、6例を改善した。
補助入力は7例を改善する一方、親が合格した次の4例が不合格となった。

- 「売上は含めない」が「売上は含めます」に反転した。
- 「導入を試みている」が不正活用「導入を試んでいます」になった。
- 「減少を試す」が「減少を試みます」となり、固定アンカーと一致しなかった。
- 「減少を試している」が「減らそうとしています」となり、固定アンカーと一致しなかった。

後半の言い換えを全て事実反転と扱うわけではないが、凍結した採点規則を緩めて加点しない。
通常入力にも語順変更、文中の表現、常体語尾による不合格が残る。
合計の改善だけで個別後退や意味反転を相殺しない。

v6で問題になったcold-depotの修飾節は、v7の両入力では
「利用不能が集中する寒冷depotでsmart charging pilotを先に行います」と保たれていた。
この1例の修正を、任意の主語・因果関係の意味保持の証明とは扱わない。
開発用・新規・回帰を含む保存raw489件を再採点し、凍結時の全検査・集計との一致を確認した。

## 8人の実討論と判断分離

同じ架空のSwift/Rust台帳、temperature 0、seed20261005、最大24event・2round、話者別cacheで、
親と候補を各1回実走した。両方とも独立8人、15event、1roundのすり合わせまで完走した。
この実走は後述の除外条件guardを修正する前の凍結コードで行った。

| 実測 | 親と通常入力 | v7と補助入力 |
|---|---:|---:|
| 実行時間 | 224.751秒 | 231.760秒 |
| model calls | 33 | 33 |
| Weight本文 | 19/23 | 23/23 |
| 凍結claimからの代替 | 4 | 0 |
| 語尾補正とschema修復 | 0 | 0 |
| 失格claim | 1 | 1 |
| whole discussion hard gate | fail | fail |

候補の公開文の抜粋は次のとおり。本文はWeight生成で、対案・賛同の前置きはコード合成である。

> 批判的設計者: 最初の版はSwiftのコアで早く仕上げ、将来の移植は追加課題として残します。
>
> 批判的設計者: 別の進め方として、最初からRustの共通コアとSwiftUIを使い、初期工数を増やし、将来の移植に備えます。
>
> 実証監査者: 修正案として、最初の版はSwiftのコアで早く仕上げ、将来の移植は追加課題として残します。

初期8人のraw、投票8人のraw、修復raw、最終選択は親と候補で一致した。
全判断呼出しのAdapterはNoneで、本文入力だけに台帳由来のproposalと数量補助情報が入った。
1組のfixtureでの分離確認であり、全ての将来判断の非回帰を証明するものではない。

BaseはD05を移植不要、D06を安全性確保と解釈する誤りが残った。
RUST_FIRSTの未許可D05は両runで拒否したが、ID検査の通過を全説明の事実確認とは扱わない。
話者間の本文も非常に似ており、自由な自然討論が完成したとは呼ばない。
今回の時間は候補が長く、速度改善も主張しない。

## 除外条件guardの修正と再検査

「売上は含めます」は固定アンカーで不合格だったが、当時の共通`aligned`検査はTrueだった。
実走終了後に`body_preserves_exclusions`を共通本文検査へ加え、
明示的な対象に対する「含めない」「含まない」「除外する」系の保持を確認する。
別対象を含める正常文は許可し、対象の取り違え、欠落、同一対象の矛盾も拒否する。
任意の同義語、複雑な二重否定、一般的な主語束縛を解く機能ではない。

元rawと凍結スコアは変更せず、489件を別の`*_guard.json`へ再採点した。
変更は当該1例の`aligned=True→False`だけで、直接合格数は全て不変。
悪いWeight文をコードで修正・加点したわけではなく、拒否範囲を強化した。
実MLXの当該1例再生成でも同じ反転が出て、共通検査が拒否した。
これは既知失敗の診断であり、新しい未学習問題の成功とは数えない。

実討論の保存raw15本文も修正後guardへ通し、親11/15・候補15/15で変更がないことを確認した。
保存rawの再検査であり、修正後コードで8人native討論を再実行したという意味ではない。
全98単体テスト、構文検査と差分検査が成功した。これらを自由討論全体の合格とは扱わない。

## 非昇格理由と次の課題

同じAdapterで二入力の全面安定化という目標は未達。
補助入力の個別後退、除外条件の意味反転、活用の不自然さ、Baseの根拠誤読が残っている。
通常入力だけの成功へ目標を縮めたり、失敗入力を旧Weightへ振り分ける仕組みで合格と扱わない。
次の学習は一般化した包含・除外と時制の対照例を検討するが、今回の最終問題の答えを学習へ戻さない。
安定した自然文Weightや自由な討論モデルとしての公開、既定置換は引き続き行わない。

## 保存先

```text
dataset: /Volumes/data4/cod_model_weight/datasets/claim-body-qwen35-4b-v7/mlx_dual
training: /Volumes/data4/cod_model_weight/adapters/claim-body-qwen35-4b-v7/dual
audit: /Volumes/data4/cod_model_weight/evaluations/claim-body-qwen35-4b-v7_20261005
```

`evaluation_plan.json`と`source_snapshot`に、選定規則・入力契約・sourceのSHAを保存した。
選定Weightは754,632bytesで、SHA256は`a3266c5225451ef9efa21a97169a04e818bd530e92789ffe5dae8199c1665610`。
Baseとは別の本文専用LoRAで、単独実行できる全モデルWeightではない。
Baseと親Weightの学習前後SHAも一致した。
`evaluation_audit.json`、`case_nonregression.json`、`runtime_comparison.json`、`guard_audit.json`に
全比較・個別後退・実走・修正後再採点を分けて保存した。
[追加教材](../data/general_body_qwen35_v7/additions.json)、[全教材](../data/general_body_qwen35_v7/curated.json)、
[system](../configs/claim-body-qwen35-4b-v7-system.txt)、[学習設定](../configs/claim-body-qwen35-4b-v7.yaml)を参照できる。
[非昇格記録](../promotions/qwen3.5-4b-claim-body-v7-step256.json)にSHAと運用境界を保存した。
