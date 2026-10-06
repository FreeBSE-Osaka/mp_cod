# Qwen3.5 全文発言用Weightの初期化比較

Generalの自然な発言を改善するため、Qwen3.5-4B-4bitからの新規LoRAと、前回の研究用step16からの継続学習を比較する。Baseの主張・根拠・投票にはAdapterを適用せず、全文rendererだけを学習する。新規条件の32ステップは正常終了し、継続条件の実学習を開始した。候補の生成比較は未実施で、Weightの採用・公開・既定置換は保留する。

## 初期Weight以外を揃える

両条件は同じ8人格・8発話行為・3入力形式、学習464表現、開発48表現を使う。学習topicは前回と同じ10件で、旧学習topic72発言のみを復習する。全てCodex作成の架空教材で、監修済みの実議論ではない。前回の最終評価回答を教材へ移さない。

学習率は両条件1e-4、32ステップ、16ごとに保存、最後4層のmlp.down_proj、rank4、scale8、batch1、上限768 tokensとする。前回の5e-6とは異なるため、前回からの変化を初期化だけの効果とは呼ばない。今回の2条件間では、初期Adapterだけを変える。

新規条件は`--parent-adapter`を省略し、既存MLX-LMの初期化を使う。継続条件は前回step16のSHAを検査して同じ引数へ渡す。思考なしのprompt mask、学習プロセス限定のキャッシュ上限0、20GiBの容量監視を維持する。

実MLX ChatDatasetで学習464・開発48・未学習72の計584表現を検査した。最大tokenは722・706・720で、prefixとJSON正解のマスク対象は一致し、切り捨ては0件だった。未学習72表現のモデル出力は生成していない。

新規条件の実設定は`resume_adapter_file=null`を確認し、32ステップを652.646秒で完走した。開発lossは初回1.942・step16で1.236・step32で1.067。最終区間の学習lossは1.149、学習済み対象tokensは1853、ピークメモリは16.931GBだった。開発と学習のlossは別集合の値であり、いずれも発言品質の改善を証明しない。
step16・32は各754632 bytesの実Adapterで、Baseモデルは別途必要である。step16の4層のLoRA行列更新と有限値をNumPyで確認し、GPUへ別モデルを載せていない。
継続条件は前回step16のWeight読込みを確認した。初回開発lossは1.475で、32ステップの比較は実行中である。
新規開始、既存Weightからの継続、例外時の環境復元、ログ出力失敗時のキャッシュ上限復元を含む116単体テストが通過した。

## 出力と採点の比較

既知開発入力のBaseは完全JSON48/48、直接検査24/48、事実保持も含むstudy_valid16/48。親step16は48/48・27/48・13/48である。plain32入力の新しい生成と、現在の入力に完全一致する理由付き16入力の保存済み生成から作った対照で、新しい全48生成や実討論ではない。

各条件のstep16・32を、同じ48開発入力・同じ220-token上限・temperature 0・seed・固定採点で比較する。入力単位の契約・発話行為・主張・数値・反応整合・競合主張の違反が少ない候補、study_validが多い候補、直接合格が多い候補、早いstepの順に選ぶ。同点なら新規条件を先にする。

未学習生成へ進む前提は、候補のstudy_validがBase16/48と親13/48を上回り、Baseで通った個別入力を失わず、原文確認で事実の意味変化がないこと。満たせない場合は候補をHOLDとし、未学習72入力を指示や候補選択へ流用しない。

数値検査には追加の診断も行う。全入力フィールドを数字の根拠にすると、`id=C999`が資料にない999枚という発言を通す反例があった。ID・専門観点・候補理由を除き、凍結主張と根拠本文だけで別に照合する。Baseの既知48出力では単独数値検査の差が1件あったが、その発言はD番号露出などの既存検査で既に不合格だった。Base・親ともstudy_valid合格への影響は0件。元の採点や選定基準は上書きせず、候補でもこの追加診断を確認する。数字の一致だけでは対象・否定・時点の正しさは証明できない。

JSON・語句照合の合格だけで自然な議論の完成とは扱わない。確認状態、否定、数量の対象、前案と変更理由を別途原文で点検する。旧全文18例とsource16例の実生成、保存済み採点の非回帰、Adapter除去、同条件8人実討論、多案での実際の見解変更・独立した賛同理由は、引き続き昇格に必要である。

## 保存先

```text
dataset: /Volumes/data4/cod_model_weight/datasets/general-utterance-qwen35-4b-v3/mlx_common_prompt
adapters: /Volumes/data4/cod_model_weight/adapters/general-utterance-qwen35-4b-v3
audit: /Volumes/data4/cod_model_weight/evaluations/general-utterance-qwen35-4b-v3_20261006
```

`plan.json`のSHA256は`8804914ec429f5834afe4f551419424a62e73c451f7feb8ff4fda5e4759f07d8`。準備時のソース・教材・対照・親WeightのSHAを固定し、`source_snapshot`も保存した。`token_audit.json`は入力検査、各学習のjob journalは進行・終了・資源停止の記録である。
`fresh_training.json`は新規条件の完走・親なし設定・各WeightのSHAを記録する。step16は`f72ae7f8463802c63e64d887c17c0bd9586fb4986d2082468bddc2e6b0f3d304`、step32は`f14c0245baf15dfb2238542a7bdc87101bd7f6367d88958b43fd6022a52292f1`。
`base_numeric_diagnostic.json`と`parent_numeric_diagnostic.json`は保存済み出力への追加照合で、新規の生成評価ではない。
