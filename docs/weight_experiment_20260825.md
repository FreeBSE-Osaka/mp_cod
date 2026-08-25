# Weight experiment — 2026-08-25

## Result

`Qwen3-1.7B-4bit + empirical_auditor LoRA step 8` を、決定論的ツール証拠がある場合だけ条件付き昇格した。汎用Weightとしては昇格していない。

## Environment

- MacBook Air M2 / unified memory 24GB
- MLX 0.32.1 / MLX-LM 0.31.3
- Weight・venv・cacheはGit管理外の外部ストレージに保存
- 学習データ240件: train 192 / valid 24 / frozen holdout 24

## Findings

- Qwen3.5 4Bはbatch 2でMetal OOM。batch 1は約13〜15GBで動いたが、複数LoRA設定で算術能力または出力品質が低下したため全て不採用。
- Qwen3 1.7Bはbatch 4でPeak 6.121GB、約23,392 target tokensを32 step学習できた。
- 32-step Weightは `条件付き` へモード崩壊したため不採用。
- 8-step Weightは証拠付き24問でBaseを上回ったが、証拠なし分数精度は低下した。

| Frozen holdout | Base | Step 8 |
|---|---:|---:|
| Evidence all-correct | 41.67% | 54.17% |
| Evidence semantic | 50.00% | 66.67% |
| Evidence fraction | 70.83% | 75.00% |
| Evidence strict contract | 79.17% | 75.00% |
| Raw fraction | 29.17% | 16.67% |
| Raw semantic | 0.00% | 0.00% |

## Boundary

このWeightは計算器ではない。コードが算出した件数・既約分数・仮説一致フラグを解釈し、公開討論用の人格回答へ整える専門家である。証拠なし入力は禁止する。

MLX生成にはJSON grammarがないため、全必須キーと型が正しい場合だけ、配列の上限超過を先頭2件へ正規化する。欠落キーや型不正は補完しない。
配列切り詰めで数値根拠が落ちないよう、決定論的ツールの確定値は `recommendation` へ必ず再掲する。

実運用スモーク `upper=100, condition=2, target=3, hypothesis=1/3` は、正規化後にJSON・契約・分数・仮説判定・semantic・all-correctの全項目を通過し、`8/25 = 0.32` を返した。

Adapter本体はこのGitリポジトリに含めない。公開時はモデルカードと共にHugging Faceへ分離し、この凍結評価とSHA-256を照合する。
