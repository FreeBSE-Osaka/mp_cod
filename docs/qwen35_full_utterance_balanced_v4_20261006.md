# Qwen3.5 全文発言用Weightの均等学習

Generalの対案・異議と事実保持を強化するため、8演者と8発話行為の64組に、4入力表現を均等に割り当てる。前回の32ステップは25/64組しか観測できず、対案と異議の直接合格が0/6だった。正解部分だけをlossへ掛ける修正済み学習器でBaseから256表現を一巡する。初回はMetalのメモリ不足で終了したため、入力を切らず未使用paddingだけを外す経路を最長入力で検証し、別実行を開始した。発言品質・Weight昇格・既定置換・公開は未確認である。

## 教材を均等にする

既存の10学習題材から、演者と発話行為ごとに1例、計64例を決定論的に選ぶ。選択はcase名の固定hashで行い、評価結果や正解文の良し悪しによって選び直さない。10題材全てが残り、分割を跨ぐ主張がないことも検査する。

64例それぞれに、理由付き・柔軟plain・旧structured・誤った候補理由付きの4表現を生成する。各表現は64件、各演者と各発話行為は32表現、各演者と発話行為の組は4表現ずつである。誤った理由付きでも根拠本文と正解文は変えず、資料の確認状態・数量・範囲を優先するよう学習する。

旧v1の追加復習は今回の均等epochには混ぜない。旧形式自体はstructured表現として全64組へ含め、過去の保存済み採点と旧全文18例などの非回帰を別途確認する。過去Weightの正解を新しい評価回答へ移すことはしない。全教材はCodex作成の架空例で、監修済み実議論ではない。

`tools/general_utterance_training.py build --balanced-pairs --profiles source_grounded flexible_plain structured_plain`で生成できる。3形式の不足、組の欠落、題材を消す選択、追加rehearsalとの併用は拒否する。既存の均等化なしの生成方法は維持する。

実MLX処理で、学習256・開発48・未学習72の計376表現を検査した。最大tokenは722・706・720で上限768以内、prefixとJSON正解spanは一致した。loss対象はpromptとpaddingを除き、EOSを残す正解位置だけである。学習の正解対象は全体で14,582 tokens。未学習72表現のモデル出力は生成していない。

## 一巡の観測を確認する

インストール済みMLX-LMのbatch1順序を事前再現した。学習中は標準callbackの対象token数と照合し、256ステップの完走時には全256表現を観測したことを確認する。以下は完了した学習の結果ではなく、同じデータとseedから得た予定値である。

| step | 観測する演者と発話行為の組 | 入力表現 | 正解対象tokens |
| --- | --- | --- | --- |
| 64 | 42/64 | 64/256 | 3620 |
| 128 | 59/64 | 128/256 | 7293 |
| 192 | 63/64 | 192/256 | 10942 |
| 256 | 64/64 | 256/256 | 14582 |

## 学習と選定条件

Qwen3.5-4B-4bitから新規LoRAを開始し、最後4層のmlp.down_proj、rank4、scale8、learning rate1e-4、batch1、256ステップとする。64ごとにWeightを保存する。思考なしのprompt mask、勾配checkpoint、学習プロセス限定cache上限0、20GiBの容量監視を維持する。Baseの主張・根拠・投票にはAdapterを使わない。

実設定は親なし・256ステップ・上限768を確認し、16ステップの勾配更新まで正常に進んだ。標準callbackによる実観測は学習対象892 tokens、学習loss1.360、ピーク15.974GB。初回の開発batch lossは1.404で、開発48入力全体の生成品質ではない。学習完走と一巡の確認はまだ未完了である。

初回は32ステップの報告で正解対象1822 tokens、ピーク16.016GBまで進んだ後、MetalのCommand buffer OOMで失敗した。終了コード1・777.171秒、失敗時ピーク17.056GBで、最初の64ステップ保存前のため学習済みcheckpointはない。プロセス終了と容量の余裕を確認し、同じ条件の再実行、他アプリの終了、OSのwired設定変更は行っていない。

均等化の追加後、教材入力と生成評価の8関数のASTが学習前のv3ソースと同一で、core validatorも変更していないことを確認した。均等化・分割保持・4表現配分を含む118単体テストが通過した。これは新Weightの発言成功を示すものではない。

## 入力を切らずメモリ使用を抑える

共有学習器へ任意の`--exact-training-padding`を追加した。標準iteratorが作ったbatchから、最長の実列を越える未使用tailだけを除く。短い列に必要なbatch内paddingと元の長さ情報は維持する。prompt、資料、話者、前案、正解、EOSを切らず、学習層・rank・batch数も減らさない。既定経路は変更していない。

全256表現を実MLXのiteratorで検査し、実token列の集合・件数・正解対象14582 tokensが元と一致した。最長722-token列は、旧batchでは737 tokensだったが、新経路では実722 tokensのまま。これはbitwiseの勾配一致や全epochの安定性を証明するものではない。

その最長列だけを使うQwen3.5-4Bの1ステップ確認は、正解71 tokensを保持し、ピーク16.548GBで更新・保存まで成功した。これは異なる実行時点の初回失敗ピークとの差であり、pairedなメモリ改善量や発言品質の改善とは数えない。

再開は同じ教材・順序・全64組・全学習層・256ステップをBaseから開始し、最長列の確認用Weightは親に使わない。別出力先`train256_exact`を使う。16ごとに救済用checkpointを追加保存するが、比較候補は元の64・128・192・256の4つだけで、選定基準も変えない。保存前の失敗で失う作業量を小さくするための変更である。

実token保持・長さ情報・native引数の維持・任意オプションの接続を含む119単体テストが通過した。全epochをこの経路で完走するかは実行中に確認する。

再開の実学習は64/256ステップまで進み、正解対象3620 tokensで予定観測と一致した。学習loss0.902、開発8 batchのloss1.172、ピーク16.433GB。64ステップのWeight本体は保存済みでSHA256は`e5d836448577c6e1c72b7f8a04228e864f1a5b111a3f26d43b06e80ffdc62a50`である。全epochの完走、候補選定、発言の品質はまだ未確認である。後から追加した秘書補佐レポートと会議調整の評価台帳は別ファイルで、凍結したcore・人格・共有学習器・全文学習器のSHAはこの計画と一致している。

前回の研究用継続32は比較参照として残すが、今回は学習親に使わない。教材配分、正しいloss境界、学習量の複合変更であり、前回との差を初期化だけ・maskだけの効果と呼ばない。

保存候補を、同じ48開発入力・220-token上限・temperature 0・seedで比較する。今回の選定は、Baseのstudy_valid合格を失った入力が少ない候補、重大違反が少ない候補、study_validが多い候補、直接合格が多い候補、早いstepの順に固定した。前回の選定やrawは変更しない。

未学習生成へ進む前提はstudy_validが20/48を上回り、Baseの合格16入力を失わず、対案と異議の直接検査がそれぞれ4/6以上で、原文の事実・自然さにも問題がないこと。語句検査の成功だけで意味の正しさを断定しない。基準未達なら候補をHOLDとし、未学習回答を候補選択へ使わない。

旧全文18例・source16例の実生成、保存済み採点の非回帰、Adapter除去、8人の同条件実討論、多案での実際の修正と独立した同意理由は、引き続き昇格に必要である。学習完走やloss改善だけではGeneralの完成とは扱わない。

## 保存先

```text
dataset: /Volumes/data4/cod_model_weight/datasets/general-utterance-qwen35-4b-v4/mlx_balanced_epoch
adapters: /Volumes/data4/cod_model_weight/adapters/general-utterance-qwen35-4b-v4
audit: /Volumes/data4/cod_model_weight/evaluations/general-utterance-qwen35-4b-v4_20261006
```

`plan.json`のSHA256は`c18603d8e0427285f347957cd40d81e9aa15fd994984853f9ecb452301e645d4`。学習前にソース・教材・比較参照・選定条件を固定し、ソースsnapshotも保存した。
`token_and_coverage_audit.json`は入力と予定順序の監査、`native_training_progress.json`は実学習callbackの観測、job journalは終了・失敗・資源停止の記録である。native観測はモデル発言の評価ではない。
初回の`train256.job.json`とprogressは失敗記録として保存し、上書きしていない。`memory_pivot_exact_padding.json`は入力保持と最長列の監査、`max_length_exact_smoke1.job.json`は最長列の実更新確認である。
再開条件は`plan_exact.json`へ固定し、SHA256は`39dcf74b1747d0e8b5836efc502e1e3e2b30ca9043c9d98074eda532dae54ab3`。更新ソースは`source_snapshot_exact`へ保存した。再開のnative観測は`native_training_progress_exact.json`、完走と候補SHAは`training_exact.json`へ記録する。
