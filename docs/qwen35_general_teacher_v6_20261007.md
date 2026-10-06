# Generalの教師教材候補と学習前の点検

2026年10月7日、既存のQwen3.8 27Bモデルから、Generalの8演者・8発話行為に対応する教師教材候補64件を生成した。元の26件を保持し、残る38件を資料レビューに基づいて修正して、全64組の研究用教材を準備した。元の320学習入力へ128入力を追加した448例は、全文と終端を保持して768 tokens以内に収まる。教師文を含む末尾だけに語彙出力を計算し、学習プロセス内のコンパイルを無効にする方式が16 stepを正常終了した。OSメモリの起動条件を満たしたため、同じ設定で448-step本学習を開始した。学習品質は未検証で、Generalは研究HOLD、既定Weightの置換とWeight公開は行っていない。

## 学習用入力だけを教師へ渡す

既存の`general_utterance_qwen35_v2`の学習用10テーマから、`balanced_training_cases`で各演者・発話行為の組を1件ずつ選んだ。選定ケース64件と組64個は一意で、各演者・各発話行為は8件ずつある。発話行為は提案、対案、指摘、同意、改善、補足、維持、見解変更である。

教師へ渡したのは実行時と同じsource-grounded入力のsystemとuserだけで、元の正解文、採点用の必須・禁止regexは渡していない。開発評価48入力と未学習72入力も教師へ送っていない。前のv5の評価用ファイルについて、生成後もvalid・testのSHAがmanifestと一致することを確認した。

教師はローカルの`hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q2_K_XL`を使った。Ollama、JSON形式、context2048、最大320生成tokens、temperature0、seed20261008、think=falseで1件ずつ生成した。320 tokensは上限であり、使用量の目標ではない。追加ダウンロードやBase変更は行っていない。

## 生成結果と検査の区別

64件の生成jobは3284.329秒、終了コード0で完了し、全応答がstopで終了した。終了後のモデル常駐一覧が空であることを確認した。生成時の未承認原文と点数は保持し、後述の教材採用は別記録とした。

| 確認項目 | 件数 |
| --- | --- |
| JSON契約を満たす候補 | 64/64 |
| 現行の直接描画検査を通る候補 | 43/64 |
| 固定の資料語句検査まで通る候補 | 36/64 |
| 原文レビューで資料の意味を保つ候補 | 45/64 |
| 原文レビューで修正を要する候補 | 19/64 |

これらは異なる確認項目であり、一般的な正答率ではない。語句の検査を通っても、上限を実費にする、比較の上限を実測件数にする、原文にない可逆性を保証する等の問題があった。一方、意味を保った自然な文でも、現行の発話行為markerや字面の必須語と一致せず拒否される例があった。

資料にはない「すぐ戻せる安全な設計」、費用上限からの「その金額以内で済む」という保証、18枚の比較上限からの「18枚の比較結果」、比較の提案から採用への変更は修正対象とした。主張の意味、測定した対象、確認・未確認、上限と実績を混同させない。

「合意」「別案」の自然な表現、否定の言い換え、未定と未決定等は、原文レビューと現行検査の不一致として分けて記録した。点数を合わせるために旧検査や原文を書き換えていない。演者の価値判断と、観察された事実や保証も区別して点検した。

## 768-token制約

Qwen3.5-4Bの既存トークナイザーと、学習時のChatDataset・assistant-only maskを使ってCPUで数えた。モデルをロードした計算や学習ではない。入力と教師のJSON出力を合わせて63件は768 tokens以内、1件は777 tokensだった。

JSONの外側の余分な空白を標準化する別監査でも、その1件は777 tokensのままだった。IDと発言本文が変わらないことを確認し、原文は保持した。その候補は、条件と未確認状態を保つ別の編集修正文へ差し替える研究教材とした。原文の切り詰めや最大長の引き上げは行っていない。

## 原文の不足と修正教材

64件の原文と4つのレビュー記録を、rawとrequestのSHAで照合した。資料の意味、現行の固定検査、768-token制約をすべて満たす原文は26件、演者・発話行為の組も26個だった。残る38組を補う前に、独立した再生成指示を6入力で試したが、固定資料検査の合格は3/6に留まり、上限や未確認条件を落とすため一括適用しなかった。

38件はCodexが資料と凍結主張を確認して編集修正した。独立教師生成ではないため、`source_review_editor_correction`という来歴を持たせ、保持した26原文の`teacher_raw_retained`と区別する。全64件は既存検査を通り、元入力と本人限定修復入力の128表現は最大758 tokensだった。この確認は教材の適合性であり、未学習入力でのモデル性能ではない。

研究用SFTは元320入力をバイト単位で保持して128入力を追記し、448入力にした。各演者・発話行為の組は7表現ずつで、開発48入力と最終72入力は元ファイルとバイト単位で一致する。元候補の未承認metadataは書き換えず、`adoption_review.json`と新教材manifestへ研究学習の許可を記録した。製品への昇格許可はfalseのままである。

## メモリを抑える学習検証

CPUの実Tokenizer・native iterator・assistant-only lossの確認では、448入力の教師token数は28,217で、EOSと末尾改行を保持する。標準paddingの入力形状は12種類で、実長ごとの209種類より少ない。ただし形状数だけではメモリ削減を証明できない。

標準padding、batch1、4層・rank4、cache上限0、勾配checkpointの16-step試験は、最初の学習報告前にMetal OOMで終了した。MLXピークは17.058GB、OSの観測physical footprintピークは約17.8GiBであり、別の指標である。終了コード1と対象プロセスの終了を確認し、448-step本学習へは進めなかった。

別方式では、全文の文脈処理は維持し、既存lossで重みがゼロのプロンプト部分に語彙出力を作らない。固定した末尾160 tokensに、全train・validの教師位置が未使用paddingを含めて収まることをCPUで照合した。教師tokenの削除、文脈の短縮、モデル層の削除ではない。小さなCPUモデルの6比較で、従来lossと新方式のtoken数・loss・勾配を照合し、最大勾配差は約1.5e-8だった。

最初の別方式試験は、CPU検証モジュールのimportがdefault deviceをCPUへ変える不具合により中止した。CPU設定を検証main内だけへ移したGPU試験は、実Qwenの同一入力でlossが両方式とも0.610329449、教師token数が両方58であることを確認した。これは1入力のloss比較であり、実Qwen全体の勾配のビット一致を検証したものではない。

GPU試験は16 step、1,052教師tokensを処理し、Adapter保存とプロセス終了まで終了コード0で完了した。所要422.271秒、MLXピーク16.640GB、OSの観測physical footprintピーク約19.3GiBだった。最終train loss0.899、検証2 batchのloss0.939は、発言品質の合格判定ではない。最適化は外付けの研究コードだけで、共通製品経路は変更していない。

本学習の起動条件は、短時間試験が正常終了し、観測physical footprintピーク18GiB以下であることとした。20GiBの実行停止線まで2GiBの余裕を残すため、19.3GiBの候補では448-step学習を起動しないことを実際に確認した。

その後、同じ16入力の順序・親・学習設定で、学習プロセス内だけのコンパイルを無効にした。16 step、1,052教師tokens、保存・終了まで終了コード0で完了した。所要332.905秒、MLXピーク12.241GB、OSの観測physical footprintピーク11.773GiBで、18GiBの起動条件を満たした。前のコンパイル有効版の422.271秒・19.252GiBとの短時間比較であり、一般的な速度改善や長時間運用の保証ではない。OS全体のwired設定や他アプリは変更していない。

コンパイル無効版でも、実Qwenの同一入力で従来lossと末尾投影lossはともに0.599083066、教師token数は58だった。ただしコンパイル有効版の0.610329449とは異なり、両モードの計算結果や更新Weightが完全同一とは主張しない。最終train lossも有効版0.898789・無効版0.904694で、どちらも発言品質の合格判定ではない。

448-step本学習はコンパイル無効版で開始した。最初の16-step実報告は教師1,052 tokensで予定値と一致した。実行中もOSメモリ20GiB、内蔵と外付け双方の空き容量20GiBを監視し、対象workerだけを停止できるようにする。教材、4層・rank4、学習率、思考なし、全文の文脈と教師位置、候補の選定規則は変更していない。

学習完了後は、固定した開発候補112・224・336・448から事前の非回帰規則で選ぶ。未学習72入力を候補選びに使わず、Base・親の個別合格の保持、旧生成検査、Adapter切り離し、全8人討論での意味保持を別に確認する。`evaluate_all_eager.py`は学習の終了記録を確認するまで評価を起動せず、`select_eager.py`は固定した48入力から選ぶ。短時間学習の成功だけでWeightを昇格させない。

## 再現記録

```text
/Volumes/data4/cod_model_weight/evaluations/general-teacher-utterance-v6_20261007
  plan.json
  source_snapshot
  selected_training_inputs.json
  candidates.json
  candidates.json.job.json
  quality_profile_complete.json
  token_audit_complete.json
  canonical_token_audit_partial.json
  manual_review_batch01.json
  manual_review_batch02.json
  manual_review_batch03.json
  manual_review_batch04.json
  review_complete.json
  repairs/curated_candidates_v1.json
  repairs/curated_token_audit_v1.json

/Volumes/data4/cod_model_weight/evaluations/general-utterance-qwen35-4b-v6_20261007
  plan.json
  adoption_review.json
  token_and_coverage_audit.json
  smoke16_memory.json
  tail_projection_cpu_check_v2.json
  tail_projection_plan_v2.json
  smoke16_tail160_gpu_loss_equivalence.json
  smoke16_tail160_gpu_progress.json
  smoke16_tail160_gpu_training.json
  smoke16_tail160_gpu_memory.json
  eager_plan.json
  compiled_eager_pilot_comparison.json
  smoke16_tail160_gpu_eager_training.json
  smoke16_tail160_gpu_eager_memory.json
  train448_tail160_gpu_eager_progress.json
  train448_tail160_gpu_eager_memory.json
  evaluate_all_eager.py
  select_eager.py

/Volumes/data4/cod_model_weight/datasets/general-utterance-qwen35-4b-v6/mlx_curated_epoch
  train.jsonl
  valid.jsonl
  test.jsonl
```

教師原文のSHA256は`f01d65c2b5fe517c06cafa7d87eefee35ee6ffea4a287ce955f169c4bc741855`、全件レビューのSHA256は`e68ec132fdae52590aceeb42f0596e9bec92b0b40b95abfdee081d7a8e168622`。生成・点検scriptと生データは外付けへ保存し、公開するのは検証範囲の説明だけである。
