# General Dialogue utterance renderer v4 — 2026-09-03

## Result

v3の未学習move混同を、到達可能なpersona×moveの均等化とtopic分離で再学習した。共有step160 WeightはBaseより機能transferしたが、別topicでの直接合格率が十分でなく、**非昇格**とした。

- `promotion_allowed=false`
- `parent_replacement_allowed=false`
- `automatic_publish_allowed=false`
- Base本体と構造判断は固定
- Weight、dataset、評価rawはdata4に置き、Gitには含めない

## Validator repair

学習前に、自然な正解を誤失格にしていた共有validatorを修正した。

- `賛同`、`不十分`、`異議`、`危険`、`見方は同じ`等を発話行為として認識
- 長い自然文は、選択claimの方向付きn-gram被覆と競合claimとの差で照合
- object文は対象案を説明してもよいが、対象案を「選ぶ・採る・支持する」文は拒否
- 新規提案が`その案に賛成`、維持、変更を装う場合は拒否
- 短い賛同をsanitizerで二重化せず、異議文の`その案には…代わりに、その案には…`を除去

29 unit testsに自然な賛同、異議＋代案、方向付きclaim照合、競合案選択、propose偽装、sanitizer重複を追加した。

## Reachable move matrix

全人格×全moveを機械的に作らず、event scheduler上で到達可能な組だけを対象にした。

| Persona | propose | object | agree | maintain | revise |
|---|---:|---:|---:|---:|---:|
| hypothesis_builder | 6+ | 6 | 6 | 6 | 6 |
| falsifier | 6+ | 6 | 6 | 6 | 6 |
| empirical_auditor | 6+ | 到達不能 | 6 | 6 | 6 |
| pragmatic_operator | 6+ | 到達不能 | 6 | 6 | 6 |

実証・実行personaのclaim catalogには対立codeがないため、event `object`を捏造していない。

## Frozen data

- train topic: autoscaling、backup、cold storage、inventory、invoice OCR、reply assist
- valid/test topic: email triage
- functional unseen topic: EV fleet
- 承認済み自然文: 95
- 決定論的move補完: 58
- train / valid / test: 153 / 8 / 8
- 最大token長: 552
- move補完は既存claim labelだけを使い、新しい事実・数値を生成しない
- Qwen3.5 teacher候補はmove遵守0が続いたため途中停止し、全件を学習から除外

```text
dataset manifest  9cd7784ed63b4c3262efbafdcc4d76839b28d870b16d9995043d07c533da918c
augmentation       d582f193440ad1b9c988762c5b3153221b0d03d0c60b55fe247449893bf9a676
functional unseen  96eca076f6569bd65b56afd5561fc2cdbe170cc5d3517d8eb40f2119fdddffda
```

## Training

設定は [`configs/general-dialogue-v4.yaml`](../configs/general-dialogue-v4.yaml)。Qwen3-1.7B MLX 4bitをBaseに、LoRA rank 8 / scale 16 / last 8 layers、learning rate `2e-5`、max sequence 640で学習した。

| Candidate | Valid loss | Test loss | Functional result |
|---|---:|---:|---|
| shared step80 | 0.769 | 未計測 | EV 9/18、競合案選択1 |
| shared step160 | 0.312 | 0.232 | EV 14/18、email 5/16、競合案選択0 |
| shared targeted step184 | 0.266 | 0.191 | step160失敗3件が0/3へ退行 |
| move-balanced step60 | 0.878 | 0.590 | EV 5/18、schema 12/18、反復崩壊 |
| falsifier persona step48 | 0.913 | 0.336 | EV 1/5 |
| falsifier persona step96 | 0.670 | 0.102 | EV 3/5 |

loss低下とschema学習は再現したが、未学習topicの発話行為・claim保持とは一致しなかった。残り3人格の個別学習は、批判役pilotが5/5へ届かなかった時点で停止した。

## Best Weight boundary

共有step160の厳格評価:

```text
EV fleet: Base 0/18 -> Weight 14/18
email:    Base 0/16 -> Weight 5/16
EV pipeline: direct 14 / sanitized 1 / fallback 3
competing claim selections: 0
mechanical utterances: 0
```

EVでは改善したが、別topicのemailで5/16に留まり、実走4 eventでもWeight由来は1件だけだった。よって実用Weight、昇格Weight、配布候補とは呼ばない。

## Runtime fallback improvement

未昇格Weightへ依存せず、無効なobject / agree / maintain / reviseを複数の検証済みtemplateへ戻すようにした。templateは人格・イベント順でローテーションし、学習export対象にはしない。

Adapterなし`--fast`のEV 4-event実走:

```text
model calls: 4
model claims: 8/8
rejected claims: 0
renderer calls: 0
elapsed: 23.2 seconds
```

異議イベントは次のように表示された。

```text
仮説構築者: その結論には異議があります。まず『電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開する』で条件を確かめる案を提案します。
```

これは表示改善でありWeight成功ではないため、`template_fallback`として記録しhard gateは通さない。

## Hashes

各Adapter Weightは約9.6MB。

```text
config v4                    a3e9243c3bf9d83fd6c0a258e3cb8e87d2677c9de0d0fcce688b715d791ba96f
shared step80 weights        d82a1e92109c9e83e596f1fe2580f68466757dbbf4279a79eb99c89d28d443fc
shared step160 config        608b6c9b5251ce3fd3807863bb3bf189dedcbdfbc597622ae4fcf2163f725ae9
shared step160 weights       06c221491f41c40ce19a6922236e60eaa1d10e03a8f502d078d726bc091217ab
shared step184 weights       dde41ccb125e6a30f67d9c5b2a3663183892e04681b71833e56036eea74bfb90
move-balanced step60 weights da43ac7b1444d3c8c97c71c2dbd73bc2f82bed660bfc241051420c7a0112aa0f
falsifier step96 weights     eab4444371781d724cc60586b6e3b495c73b2f2e480027fc90d74341d224a39f
EV step160 evaluation        4ae4494b8c561f85047bbbf3206343b491eb6208cd758b4c9b3e7d91a222610c
email step160 evaluation     9ff84a57c30cbf5ef8fe75251cf2b80180592b9fd819749372c7da36b8419b78
```

## Stop condition

追加iteration、rank拡大、残り人格の個別学習は行わない。次に再開する条件は、別topicで自然な実会話targetを増やし、EVとemailの両方で直接valid率を改善し、競合案選択0、Base構造判断の完全一致を同時に満たせることとする。
