# General Dialogue natural specialists v6 — 2026-09-04

## Result

Qwen3-14Bを1発言ずつ使い、異なる12 topicから`hypothesis_builder/event object`を12件、`pragmatic_operator/event agree`を13件へ増やした。人格・phase・moveを分離してLoRAとrepair LoRAを評価したが、未学習holdoutは親を上回らず、**全候補を非昇格**とした。

- `promotion_allowed=false`
- `parent_replacement_allowed=false`
- `automatic_publish_allowed=false`
- Baseのclaim・evidence・vote判断は変更しない
- Weight、teacher raw、dataset、評価rawはdata4に置き、Gitには含めない

## Natural target collection

12の未使用General ledgerをtopic単位で使用した。Qwen3-14Bは1 call 1発言、厳格JSON schema、object/agree専用指示で実行し、各rawを逐次保存した。

- 初回3 topic: 4/6 direct valid
- 全12 topic自動判定: hypothesis 8/12、pragmatic 11/12
- 手動意味監査: route optimizerの対象案への寝返りと無断`早急`を追加失格
- v5の直接合格、bike smoke、凍結claimだけを使った手動整文2件を統合
- 最終: hypothesis 12、pragmatic 13

手動整文2件は`user review required`として記録し、昇格根拠には使わない。

```text
curated targets SHA256 c8b778b41a3aaf99f51ff2327ca205dcc23206d01b42286f4b37237d245cd2bd
hypothesis manifest    296d65261fbf23f4725785d9a7c57f5101af96f1c4fe0074cd33319cb8d31c77
pragmatic manifest     8b642a5e8c4c323317eaa9b01f1560c84e9791b6de02dd8f461e50ff6ad33405
```

## Number grounding guard

Qwen出力が入力にない`12か月 / 18か月`を創作したため、renderer payloadに存在しない数字を拒否する`dialogue_numbers_are_grounded`を追加した。NFKC正規化後の数値集合を比較し、claim・target・evidenceにない数字が一つでもあればfallbackする。

既存29 unit testsへ、groundedな`30 / 240`の許可と、未入力`12 / 18`の拒否を追加した。

## Specialist datasets

| Specialist | Train | Valid | Test | Holdout |
|---|---:|---:|---:|---|
| hypothesis / event object | 10 | 1 | 1 | subscription、experiment、EV、email |
| pragmatic / event agree | 11 | 1 | 1 | subscription、experiment、EV、bike |

subscriptionとexperimentを学習から除外し、EV/emailも完全未学習のまま保持した。

## Direct specialist results

| Candidate | Training | Parent | Candidate |
|---|---|---:|---:|
| hypothesis step16 | shared step160 + 16 iter, LR 2e-6 | 0/4 | 0/4 |
| hypothesis step32 | shared step160 + 32 iter, LR 1e-5 | 0/4 | 0/4 |
| pragmatic step16 | shared step160 + 16 iter, LR 2e-6 | 2/4 | 2/4 |
| pragmatic Base step48 | Base + 48 iter, rank 4, LR 2e-5 | 2/4 | 1/4 |

仮説step32はvalid/test lossが0.778/0.996まで下がったが、holdoutでは内部protocol復唱、賛同への反転、claim欠落が残った。実行Base specialistも内容は出したが、賛同moveを安定して付けられなかった。

## Repair specialist

実行agreeの正解11件を、次の3種類の拒否文から修復するdatasetへ変換した。

- 賛同だけでown claimがない
- own claimだけで賛同がない
- 賛同も追加条件も曖昧

Base repair step48はvalid/test loss 1.718/1.253だった。親が失敗したexperimentとEVへ適用したが、失敗文をそのまま複製して0/2。Qwen3.5-4B Base repairもEV本文は意味的に良好だったがJSON契約を欠き、experimentでは未入力の12/18か月を再創作して0/2だった。

```text
repair dataset manifest 44ae71bc7f1ab911d1101e30b60f501c4181c5dceefbb86870fdbf21c72f0d43
repair evaluation       e48fe05251b0d1785a958eb7c066512089d2a0bd38b205461c51cd66b91ee4b4
4B repair evaluation    b809106b845493b97782b1fd34e2c8b26754b4fc0df602abe25f92a05c3351d8
```

## Weight hashes

```text
hypothesis step16 weights        1fc20894becb91c99f738ead9ce1f325ad82ee4f63c98701dc62c3cdf7628ba5
hypothesis step32 weights        ac87dc23b148c1f875f1c3483b804dfe61241e7ad99085c698e356a4ec9c84d6
pragmatic step16 weights         5d814a0f704d7990b07f68c5d66aa5170c0935394e6f451cead578deb29be6f9
pragmatic Base step48 weights    51a7f9926a2d6431a8ca452912c7c7056cccf7131a3845a1575efefca6a8e0c9
pragmatic repair step48 weights  fc17c4491da4d7334ce1d1b466b981f2255a1840ddfbf79198040c3aa2a304c8
hypothesis step32 evaluation     7618c4a4c33d652ac3e248a2f1737ef833720e52e519b969ec3fae8a62a20ab8
pragmatic step16 evaluation      78acc871570983e10a64985ed694b98a32ebae738e8af1685b39b4c7a6165051
pragmatic Base step48 evaluation e25e25dcd5e426495476f317a437aa13017b2e5f64cd6a40932ff6b30881d0c8
```

## Stop condition

異なるtopicを12件まで増やしても、1.7B LoRAは発話行為を別topicへ移せなかった。追加iteration、repeat、repair LoRA、4B repairは行わない。現行運用はWeightなしBase構造判断と検証済みmove別fallbackを維持する。

次の研究候補は生成LoRAではなく、move markerをコードで確定して自然なclaim説明だけをモデルへ任せる構成、または出力制約を強制できるruntimeである。これは現在の安全fallbackを壊さず別branchで評価する。
