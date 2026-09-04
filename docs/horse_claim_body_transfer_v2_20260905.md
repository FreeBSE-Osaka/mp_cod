# Horse Claim Body transfer v2 — 2026-09-05

## Result

昇格済みClaim Body v3を、学習から除外された競馬システム改善ledgerへ適用した。Baseはclaim、evidence、confidence、voteを担当し、Weightは凍結claim本文だけを描画した。

- event 8 + reconciliation 12 = public body 20/20
- body renderer model call 6、cache hit 14
- reaction failure 0
- competing claim selection 0
- summary conflict-free true
- hard gate pass
- elapsed 94.7秒
- horse専用Adapter追加学習: 不要

Weightの昇格範囲はclaim本文rendererのままであり、競馬予測や投票の正しさを意味しない。

## Shared fixes found by horse transfer

### Global conflict coherence

局所pair投票で`A > B`と`C > A`が同時成立すると、旧summaryはAとCを同時consensusへ入れ得た。別論点で敗れた局所winnerは全体winnerにせず、そのpairをunresolvedへ戻すようにした。循環や連鎖を推測で決着させない。

`event_run_metrics`へ`summary_conflict_free`を追加し、矛盾summaryはhard gateを通さない。

### Frozen-claim-aware validation

- 凍結claimに存在するunderscore tokenだけBodyとmove validationで許可する。
- 生成側が追加した未知tokenは従来どおり拒否する。
- claim本文中で話題にしている「異議・賛同・変更」はpropose偽装として扱わない。
- claim中の回数を超えて追加された「賛同」等は、提案を装った応答として拒否する。
- 同じclaim-aware検証をBody合成だけでなく、共有rendererとsanitizerにも適用する。

これにより、`positive_attention_only`と自然文moveを話題にする競馬claimの安全なrawを、再学習なしで正しく採用できた。

## Independent judgment boundary

別の最新ledgerでは本文16/16、hard gate passだったが、Base投票は回顧8/30改善だけを根拠に本番BOX置換を選んだ。hard gateは出力契約と証拠参照の合格であり、意思決定の妥当性ではない。horse側Codex監査は、validation overlap劣後と前向き標本不足を理由に置換を却下した。

## Verification

- `python3.11 -m py_compile cod_model.py test_cod_model.py`
- `python3.11 -m unittest -q test_cod_model.py`
- 32 tests OK

Horse側のモデル訓練と全CoD記録は `/Users/osaka/src/horse/docs/learning/20260905_horse_domain_model_v3/training_and_cod_report.md` にある。
