# Qwen3.5-4B utterance renderer smoke — 2026-09-04

## Result

既存のMLX 4bit Baseを、General v4 renderer datasetで低負荷LoRA smokeした。Qwen3-1.7Bより自然な異議文は生成できたが、難所3 moveの厳格合格はBaseとstep20の両方が1/3で、機能改善がなかったため停止した。

- `promotion_allowed=false`
- `parent_replacement_allowed=false`
- `automatic_publish_allowed=false`
- 40/80 iterationへの延長なし
- 既存horse renderer Weightとdirty sourceは変更していない

## Model and memory

- Base: `/Volumes/data4/cod_model_weight/models/Qwen3.5-4B-4bit`
- MLX model size: 2.9GB
- LoRA: rank 4 / scale 8 / last 4 layers
- batch: 1
- max sequence: 640
- trainable: 1.015M / 4,205.750M（0.024%）
- inference peak: 2.853GB
- gradient checkpoint smoke peak: 8.572GB
- no-checkpoint smoke peak: 10.430GB
- full-row training peak: 11.457GB

batch 1ではM2/24GB上でOOMしなかった。gradient checkpointを外してもfull trainingは`0.081 it/s`程度で、速度律速は4B本体だった。

## Frozen critical checks

General v4の未学習EVから、1.7Bで失敗した3件だけを固定した。

| Check | 4B Base | 4B step20 |
|---|---:|---:|
| hypothesis_builder / object | fail | fail |
| falsifier / object | pass | pass |
| pragmatic_operator / agree | fail | fail |
| total | 1/3 | 1/3 |

Baseとstep20の批判役objectは、全車両展開へ異議を示し、寒冷depot pilotを対案として提示できた。一方、仮説役objectは自分の全車両展開案を否定して相手のpilot案へ反転し、実行役agreeはJSON契約または賛同moveを満たさなかった。

## Training stop

- initial valid loss: 1.633
- step20 valid loss: 1.543
- step20 train loss: 1.323
- functional gain: `+0/3`

lossは下がったが機能transferは同値だった。20 iteration checkpoint保存後に停止し、残り20/60 iterationは実行していない。

## Artifacts

```text
config
/Users/osaka/src/cod_model/configs/general-dialogue-qwen35-4b-v1.yaml

adapter
/Volumes/data4/cod_model_weight/adapters/general-dialogue-qwen35-4b-v1/shared_renderer_step20

Base evaluation
/Volumes/data4/cod_model_weight/datasets/general-dialogue-v4-balanced/qwen35_4b_base_critical_eval.json

step20 evaluation
/Volumes/data4/cod_model_weight/datasets/general-dialogue-v4-balanced/qwen35_4b_step20_critical_eval.json
```

```text
config SHA256       9bc7ee93fcf44fbdb0c6997c3b5051bddd40da8645cd23b371d2a8154a33fd72
adapter config      86b43595382bc8836155be555484af8d4a1b0826b3ea3819a475d1d43be54dfc
adapter weights     15b44f087e88a8be52e91ccfc424815eb8c582f54d91d46ed9cd5ee1648b795f
Base evaluation     f99d6840db37c9579e6a3df530c05e26cc4a5c3768ea15c1a1e56a6f2c98d366
step20 evaluation   f58a634f5ab4725284350b969381bcd475d54bbe13c4bfd51d344d5e855a02e6
```

Adapter Weightは約3.9MBで実在するが、実用Weightまたは配布候補ではない。

## Next gate

4Bのiteration追加ではなく、仮説役objectと実行役agreeの別topic自然文targetを増やし、3件の短縮gateでBaseを上回った時だけ再開する。現在の推奨運用は、WeightなしBase構造判断と検証済みmove別fallbackである。
