# General Dialogue Weight v2 experiment — 2026-08-28

## Result

自然会話v2の承認済み8runから4人格の個別LoRAを学習した。全人格で凍結valid/test lossは改善したが、未学習の構造化payloadから`utterance`を生成するtransfer評価ではBaseと同じ入力JSON復唱となったため、**非昇格**とした。

- `promotion_allowed=false`
- `parent_replacement_allowed=false`
- `automatic_publish_allowed=false`
- v1 Adapterへの追加学習ではなく、別のv2 Adapterとして保存
- 追加学習は実行設計者1人格のstep16 smokeで停止

## Frozen data

承認済み8runからfallback、不正文、機械文、近似同文を除外した。1件の不自然な`「ことが判断です」`だけをmechanicalとして除外した。

| Persona | Train | Valid | Test | Total |
|---|---:|---:|---:|---:|
| empirical_auditor | 27 | 3 | 3 | 33 |
| falsifier / 批判的設計者 | 27 | 3 | 3 | 33 |
| hypothesis_builder | 27 | 3 | 3 | 33 |
| pragmatic_operator / 実行設計者 | 26 | 3 | 3 | 32 |

Frozen manifest SHA-256:

```text
181f95fe7b45b334a2593ebdca30926ce2637826c1b89e4508e33c84dfaff339
```

## Training configuration

設定は [`configs/general-dialogue-v2.yaml`](../configs/general-dialogue-v2.yaml)。

- Base: Qwen3-1.7B MLX 4bit
- LoRA rank 4、scale 8、last 4 layers
- micro batch 1、gradient accumulation 4
- 32 micro iteration = 8 optimizer update
- learning rate `1e-5`
- max sequence length 704
- trainable parameters 0.623M / 1,720.575M（0.036%）
- peak MLX memory 約1.86GB

## Frozen loss

| Persona | Initial valid | Final valid | Base test | LoRA test | Base ppl | LoRA ppl |
|---|---:|---:|---:|---:|---:|---:|
| empirical_auditor | 3.420 | 3.171 | 3.839 | 3.484 | 46.460 | 32.580 |
| falsifier | 4.111 | 3.816 | 3.961 | 3.576 | 52.494 | 35.737 |
| hypothesis_builder | 3.742 | 3.458 | 3.608 | 3.277 | 36.901 | 26.485 |
| pragmatic_operator | 3.862 | 3.581 | 3.698 | 3.345 | 40.375 | 28.374 |

## Transfer evaluation

学習に使っていないEV fleet台帳を固定し、speech exampleとfew-shotを除いた`orthogonal_bare`でBaseと4 Adapter動的swapを比較した。

Base run:

```text
shadow_score=71.15
hard_gate_pass=false
model_claims=3/12
validated_fallbacks=9
rejected_claims=9
```

LoRA runは最初の仮説構築者でBaseと同じconfidence形式違反を3件起こしたため、早期停止した。LoRAはclaim選択JSONを学習対象としていないため、event engine全体へ常時適用する構成は不適切だった。

次に、学習JSONLと同じ`phase / move / own_claim / target_claim / evidence` payloadを直接与えた。

- 出力契約を明示しない場合: Baseと4 Adapterの全てがuser JSONをそのまま復唱し、`utterance`生成は0/4。
- `JSONキーはutteranceだけ`を明示した場合: Base、step8、step16は同じ文を返し、全て句点欠落でvalidator不合格。
- 実行設計者だけ追加8 update、learning rate `5e-6`でstep16 smokeを行い、validは3.509から3.245へ下がったが、生成結果はstep8と同一だった。

よってloss改善はあるが、機能transferは確認できない。追加step、rank拡大、全人格step16は行わない。

## Adapter hashes

```text
empirical_auditor config   47a888fbd00fbd9ef6dddf5490e110daf9b736f2d6a3df3239dfa38879e2066c
empirical_auditor weights  7fad2e5065416cb979fe77026ce599f91fbc34f4cdfb054e2ab0893addb36260
falsifier config           56bc3c3024ea3ed153966b95f4b52cc0d3447b5c8c395c4bae87f27c77fef60a
falsifier weights          ba32627d706f20018f2aa5842327f67b90ad5669816658e00a7adf4772f6f0fe
hypothesis_builder config  5699569e90617189b576e6af7bbf0bf437d118b2345fd1f9b6406f241215e3bd
hypothesis_builder weights 4b89c43a7d22080ac447e580937c1cc88f6e396034d9029530f8805694cff90e
pragmatic_operator config  a5135be0dddd51e2c2e352606b3efb2df53676b5615e6cf6d2de37ee5ff65ffa
pragmatic_operator weights 2a7d3f2f007c7d9d98d158b6561a048b32d17f67098df1da1f928f8348e18040
adapter map                8e3ee3171bc54800f7994cf2799216dae5235ee40325aafcb8c8139676460e2e
```

step16実行設計者smoke:

```text
config   af5082f7aa10e42a3ea4354aea76d19175bdbd1c32107df8b4f5d3c37074008d
weights  c6feea4b8287fe519ee01751b7359289fb6bba3dfc95145e5139247fba81f6ab
```

## Boundary

4人格のLoRA Weightファイルは実在し、凍結lossも改善している。しかし未学習payloadで機能改善を示していないため、実用Weight、昇格Weight、配布候補とは呼ばない。現行productionはBase + 検証済みv2 prompt / sanitizer / hard gateを維持する。
