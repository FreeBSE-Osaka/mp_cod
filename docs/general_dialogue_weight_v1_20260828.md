# General Dialogue Weight v1 experiment — 2026-08-28

## Result

`Qwen3-1.7B-4bit` にGeneral 3人格の個別LoRAを8 optimizer updateだけ学習した。3人格とも凍結test lossは改善したが、未学習のtransfer台帳ではBaseと会話品質が同値だったため、**非昇格**とした。

- `promotion_allowed=false`
- `parent_replacement_allowed=false`
- `automatic_publish_allowed=false`
- Adapterは再現用候補として外部ストレージに保持し、Gitには含めない

## Frozen data

Dialogue export v1は承認済みrunだけから作成し、fallback、機械文、近似同文、不正文を除外した。人格ごとの時系列splitは次の通り。

| Persona | Train | Valid | Test | Total |
|---|---:|---:|---:|---:|
| empirical_auditor | 25 | 3 | 3 | 31 |
| falsifier | 28 | 3 | 3 | 34 |
| hypothesis_builder | 24 | 3 | 3 | 30 |

Frozen export manifest SHA-256:

```text
972a2e0f79da37bc299794ea1f1dbdf7a142e3cd4021c76f26852636ad80aa66
```

## Training configuration

再現設定は [`configs/general-dialogue-v1.yaml`](../configs/general-dialogue-v1.yaml)。

- fine-tune: LoRA, rank 4, scale 8, last 4 layers
- learning rate: `1e-5`
- micro batch: 1
- gradient accumulation: 4
- micro iterations: 32（optimizer update 8回）
- prompt loss: masked
- max sequence length: 704
- trainable parameters: 0.623M / 1,720.575M（0.036%）
- peak MLX memory: 約1.86GB

Base model SHA-256:

```text
model.safetensors  0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f
config.json        507a6701220524eb8b283425bf0856a9ae4f21f4052e563896ddd668994b1dc7
```

## Frozen evaluation

| Persona | Initial valid | Final valid | Base test | LoRA test | Base ppl | LoRA ppl |
|---|---:|---:|---:|---:|---:|---:|
| empirical_auditor | 3.496 | 3.244 | 3.595 | 3.238 | 36.426 | 25.493 |
| falsifier | 3.211 | 2.938 | 3.150 | 2.834 | 23.343 | 17.006 |
| hypothesis_builder | 3.356 | 3.121 | 3.292 | 2.974 | 26.902 | 19.570 |

Lossだけなら全人格で改善した。しかし未学習の予約overbooking台帳を同一seedで比較すると、Base runと3 Adapter動的swap runはともに以下だった。

```text
shadow_score=93.33
hard_gate_pass=true
mechanical_utterance_rate=0.50
model_statement_rate=1.00
model_utterance_rate=1.00
```

独立発言とすり合わせ文も実質同一で、`「現時点では」`、`「可能性を重く見ています」`、`「確かに、見落としていました」` の反復は減らなかった。原因はWeightだけでなく、学習時と推論時のfew-shot自体が同じ定型句を強く提示していたことだった。

その後のprompt v2では、人格別speech例、複数のagree/maintain/revise文、対案必須の異議、4人目の実行設計者を導入した。Qwen3-1.7B Baseのv2 holdoutは機械文率0まで改善したため、次のWeightはv2会話を新たに収集してから学習する。v1 Adapterへ追加学習はしない。

## Adapter hashes

```text
empirical_auditor config   df99323b73f1e651dc204212448e1533f3acb8534ba5164e196ca722e9f5bd1f
empirical_auditor weights  d22ddc34e25907dbeb3beecd1d0d3d96121c45bf6ff71afc518af36a48d308db
falsifier config           69bc6c4ebcd2fa0a3895c801ca80d466e691f11a7484cd388ea692328540a13b
falsifier weights          1cf2c5ce162a985fcd5311c59ca647ec5318b7d4210d7c5fe01bc0cd20815866
hypothesis_builder config  d98d72e0616503c8a4d9185ee09a9a7432ede31af8bf4ae3e5b0c39f441887ee
hypothesis_builder weights 80866380080084c3a7398ae78c54e43583c4d08c8276bf4a0a0e7b333710830d
adapter map                b0c98d6e4793f5b78d1d2ad8fec4012771d336548b0ec0b1120d29576660a7cd
```

## Boundary

この実験はAdapterファイルが作成できることと、人格別の凍結lossが下がることを示しただけである。自然対話のtransfer改善を示していないため、Weight成功、RSI成功、親Weight昇格、公開可能性の証拠にはしない。
