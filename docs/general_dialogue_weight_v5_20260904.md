# General Dialogue natural specialist v5 — 2026-09-04

## Result

General v4で不足した`hypothesis_builder/object`と`pragmatic_operator/event agree`へ、ローカルQwen3-14Bの自然文teacher、move別few-shot、move専用LoRAを順に試した。自然targetは得られたが、未学習topicへのWeight transferは改善せず、**非昇格**とした。

- `promotion_allowed=false`
- `parent_replacement_allowed=false`
- `automatic_publish_allowed=false`
- 仮説objectの専門学習は、実行agree pilotが不合格だったため未実施
- Weight、teacher raw、dataset、評価rawはdata4に置き、Gitには含めない

## Qwen3-14B teacher

未学習topicとして異常検知閾値、事前冷却、画像検査を使った。

| Target | Direct valid |
|---|---:|
| hypothesis_builder / object | 1/3 |
| pragmatic_operator / event agree | 3/3 |
| hypothesis object repair | 0/2 |

仮説objectの2件は入力にない`即時`を追加したため失格。修復では例文をそのままコピーしてclaimを落としたため失格とした。実行agreeは、pilotへの賛同後に容量・停止条件を追加できた。

```text
実行設計者: そのpilotには賛成です。実行するなら、zone単位で停止可能な段階展開と判定条件を固定しましょう。
```

teacher raw SHA256:

```text
dedc495bca44ad9359cb1aa5a2077a113305da3f6bc44b417fa8ec41e118f059
```

## Prompt-only check

General v4 shared step160へobject例とagree例を同時に渡すと、agree例がobjectへ混入した。move別に分けると実行agreeは合格し、仮説objectも意味は保持したが、批判役objectは`賛成禁止`を明示しても`その案には賛成です`を生成した。

few-shotは一部を改善したが、人格・方向ごとの汚染を解消できないためruntimeへ採用しなかった。

## Pragmatic event-agree dataset

- persona: `pragmatic_operator`
- phase: `event`
- move: `agree`
- unique train: 4
- rehearsal train: 16
- valid/test: bike-share自然文 1 / 1
- functional holdout: EV、bike-share、refund automation
- parent: General v4 shared step160

親step160のholdout結果:

```text
EV: fail
bike-share: pass
refund automation: fail
total: 1/3
```

dataset manifest SHA256:

```text
290d33b7dd8ac5cbb34576b51b827643c015aa8e7619466f11ac381d25b6566c
```

## Specialist Weight results

| Candidate | Training | Valid/Test loss | Functional |
|---|---|---|---:|
| parent continuation | shared step160 + 12 iter, LR 2e-6 | 1.586 / 1.525 | 1/3 |
| Base specialist | Base + 32 iter, rank 4, LR 2e-5 | 2.384 / 2.014 | 0/3 |

親継続はEVで短い賛同だけ、refundで制約だけを生成し、親と同じ1/3だった。Base specialistはschema・moveとも移らず0/3へ退行した。追加iterationは行わない。

## Weight hashes

```text
parent-continuation config   710c7e33487af7daff769a826257d9b04782bbbfb44196287db592799c843e2e
parent-continuation weights  e96f5eaff3464cc2dd6385f7103805725d532ecbb0020aacff1f102fa94861f4
Base-specialist config       ac0423247419cce427b12115ae9d3e4878ac7f5b950a75346e55ce112da0d5c3
Base-specialist weights      3a2f1d4720e1f7bfa8c11e33994cf082ecd7f135f11933fd927623eefbd8673f
parent evaluation            2b638a80bf8bfedf19934464b208875555c3b55d7ae4fe1c459990cab365f08f
continuation evaluation      d0c7936de0f925907a2f2bc717f4b70c0797888ad48518f7f2bc95153f2861f2
Base-specialist evaluation   c4f4f38bb30c7c07f5b4e95e1ef2ddba8900a6ce545c3b3ac37463ba5f3e13e6
```

parent-continuation Adapterは約9.6MB、Base specialistは約2.4MB。どちらも実在するが、実用Weightまたは配布候補ではない。

## Stop condition

4件のunique targetを反復しても別topicへ移らなかった。次回は同一文のrepeatやiteration追加ではなく、各対象persona×phase×moveについて、異なるtopicの直接合格自然文を最低12件まで増やしてから再評価する。それまでは検証済みmove別fallbackを使う。
