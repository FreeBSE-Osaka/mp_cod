# Generalの教師教材候補と学習前の点検

2026年10月7日、既存のQwen3.8 27Bモデルから、Generalの8演者・8発話行為に対応する教師教材候補64件を生成した。資料と意味の点検、現行検査、768-token制約を合わせて満たす候補は26件で、64組の教材一式はまだ完成していない。追加学習、SFT採用、既定Weightの置換、Weight公開は行っていない。

## 学習用入力だけを教師へ渡す

既存の`general_utterance_qwen35_v2`の学習用10テーマから、`balanced_training_cases`で各演者・発話行為の組を1件ずつ選んだ。選定ケース64件と組64個は一意で、各演者・各発話行為は8件ずつある。発話行為は提案、対案、指摘、同意、改善、補足、維持、見解変更である。

教師へ渡したのは実行時と同じsource-grounded入力のsystemとuserだけで、元の正解文、採点用の必須・禁止regexは渡していない。開発評価48入力と未学習72入力も教師へ送っていない。前のv5の評価用ファイルについて、生成後もvalid・testのSHAがmanifestと一致することを確認した。

教師はローカルの`hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q2_K_XL`を使った。Ollama、JSON形式、context2048、最大320生成tokens、temperature0、seed20261008、think=falseで1件ずつ生成した。320 tokensは上限であり、使用量の目標ではない。追加ダウンロードやBase変更は行っていない。

## 生成結果と検査の区別

64件の生成jobは3284.329秒、終了コード0で完了し、全応答がstopで終了した。終了後のモデル常駐一覧が空であることを確認した。元出力はすべて未承認のまま保存した。

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

JSONの外側の余分な空白を標準化する別監査でも、その1件は777 tokensのままだった。IDと発言本文が変わらないことを確認し、原文は保持した。収まるように発言を切り詰めたり、無断で最大長を上げたりしていない。長い候補は短く忠実に再生成する必要がある。

## 採用前条件と不足する組

64件の原文と4つのレビュー記録を、rawとrequestのSHAで照合した。資料の意味、現行の固定検査、768-token制約をすべて満たす候補は26件、演者・発話行為の組も26個だった。残る38組は再生成または検査との不一致の解決が必要であり、26件だけで教師教材の完成とはしない。

現在の`training_adoption_allowed`と`promotion_allowed`はfalseである。26件もまだSFTへ出力していない。元の学習・評価教材、元の教師出力、元の点数、旧Weight、Generalの研究HOLDは維持している。

次の工程では、資料の上限・対象・未確認状態を明示し、失敗文や他演者の文章をコピーしない入力で未完成の組を再生成する。再生成後も全64組の意味と長さを点検する。学習を始める場合は、固定された開発評価と未学習入力、Base・親の非回帰、Adapter切り離し、全8人討論での意味保持を別に確認する。

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
```

教師原文のSHA256は`f01d65c2b5fe517c06cafa7d87eefee35ee6ffea4a287ce955f169c4bc741855`、全件レビューのSHA256は`e68ec132fdae52590aceeb42f0596e9bec92b0b40b95abfdee081d7a8e168622`。生成・点検scriptと生データは外付けへ保存し、公開するのは検証範囲の説明だけである。
