# iPhone 13 Pro / A15 native CoD — 2026-09-04

## Scope

物理iPhone内で、Baseによるclaim / evidence / confidence選択、構造化すり合わせ、Claim Body v3 LoRAによる公開文生成を1 round実行する。内部CoTは保存・表示せず、固定claim code、D番号、モデルraw、validator結果だけを記録する。

```text
Qwen3-0.6B-4bit
  ├─ 4人格の盲検初期選択
  ├─ 初期多数派の票は保持
  └─ 異論側だけ再選択
        ↓ 0.6Bを解放
Qwen3-1.7B-4bit Base
  └─ 見解変更者だけchange_reason生成
        ↓ Claim Body v3 LoRAを適用
Qwen3-1.7B-4bit + LoRA
  └─ unique claim本文だけ生成、同一claimはcache
```

LoRAは全構造選択とchange reasonが確定するまでロードしない。発話move、event順、target C番号はコードで決め、他人格の自然文を次のモデルへ渡さない。

## Fixture boundary

人格差の受入試験には`fixture_kind=synthetic_balanced`の架空災害通知アプリを使う。数値・運用条件は実在のExtremeWeatherを表さず、次の3案を均衡比較するためだけのfixtureである。

- `PILOT`: 100人へpilotを拡大して再評価
- `ROLLOUT`: feature flagを維持して全利用者へ段階展開
- `DEVICE_SPLIT`: 新端末だけ有効化し旧端末は既存経路を保持

初期候補は既存mp_codと同じ`role_preferences`で人格別に限定する。全候補を同じ順に渡した試験ではQwen3-1.7B/0.6Bとも先頭候補へ寄る位置バイアスを確認したため、これは純粋な自由選択による人格差とは主張しない。

## Preserved HOLD runs

| Run | Structure model | Initial | Final | Total | Thermal | Hold reason | SHA-256 |
|---|---|---|---|---:|---|---|---|
| 実案件ledger | 1.7B | ISOLATED 4 | ISOLATED 4 | 45.708秒 | fair | 初期人格collapse | `da998e66d6a343dfcea4dd9d970f19796e3a8ddeb7b8cdafe042ee90634645cd` |
| 架空fixture・全員再投票 | 1.7B | 2/1/1 | 2/2 | 60.246秒 | serious | 長時間・未解決 | `94623e144e911f59f1d358ff84b062269e64afb0a1f35524ce6d26c01711b463` |
| 架空fixture・異論側のみ | 1.7B | 2/1/1 | 3/1 | 44.362秒 | serious | thermal | `fce94c5aedb6c5fd472a10b728c003ea75edf8d5a27cb11af6a17a967ef04da9` |
| 0.6B構造 + 1.7B本文 | 0.6B | 2/1/1 | 2/2 | 26.298秒 | fair→serious | thermal | `2a9ae413aa6c7eae40bf5c11cfcaedda86774ee0173b670cc5c15479fb4ed2c5` |

HOLD rawは`/Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_native_cod_*_20260904.json`へ保存した。

## Observed discussion

0.6B構造 + 1.7B本文のHOLD runでは、次の会話まで生成できた。最終2対2を3対1へ強制せず、未解決として保持した。

```text
批判的設計者:
100人へpilotを拡大し修正率と電池影響を再評価します。

仮説構築者:
feature flagを維持したまま全利用者へ段階展開します。

実証監査者:
100人へpilotを拡大し修正率と電池影響を再評価します。

実行設計者:
新しい端末のみ自動要約機能を有効化します。旧端末については既存経路を継続します。

実行設計者:
その結論には異議があります。新しい端末のみ自動要約機能を有効化します。旧端末については既存経路を継続します。

仮説構築者:
考え直しました。新しい端末のみ自動要約機能を有効化します。旧端末については既存経路を継続します。

批判的設計者:
私もその案に賛成です。100人へpilotを拡大し修正率と電池影響を再評価します。
```

## Accepted warm-cache run

同じ実機で生成・検証済みの3 claim本文を、Adapter SHA・claim label・raw・body・cache全体digest付きでDocumentsへ保存した。次回runでは全entryを再検証し、改変・Weight SHA不一致・claim不一致なら使用しない。これにより1.7B本文生成3 callを省き、0.6B構造選択と1.7B Baseの変更理由だけを実行した。

| Metric | Result |
|---|---:|
| Hard gate | pass |
| Start / end thermal | fair / fair |
| Blind choices | 3 claim / 4 personas |
| Structural model calls | 7（initial 4、reconciliation 2、change reason 1） |
| Structural repair / fallback | 0 / 0 |
| Public events | 7 |
| Body model calls / cache hits | 0 / 7 |
| Body fallback | 0 |
| Outcome | unresolved tie（PILOT 2 / DEVICE_SPLIT 2） |
| Structure Base load | 1.902秒 |
| Body Base load | 2.098秒 |
| Total | 17.245秒 |
| Peak task footprint | 1,333.909 MiB |
| Minimum memory-limit headroom | 2,053.169 MiB |

開始`fair`は、必要な全claimの永続cacheが再検証を通り、本文Adapter loadが不要な場合だけ許可する。cache不足時は開始`nominal`を必須とする。

保存済みcacheファイルを直接読み直した2回目もhard gateを通過した。会話、初期・最終票、変更理由、未解決結論、cache snapshotは初回PASSと完全一致した。

| Repeat metric | Result |
|---|---:|
| Total | 18.771秒 |
| Peak task footprint | 1,318.144 MiB |
| Minimum memory-limit headroom | 2,068.278 MiB |
| Start / end thermal | fair / fair |

```text
accepted run JSON  /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_native_cod_20260904.json
accepted run SHA   b38cbcaf46deaaf2d9309149849261daf463c98e882e5bfb6493cebdc47c2d75
body cache JSON    /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_claim_body_cache_20260904.json
body cache SHA     c9aa466ce06385647e8087f184e84bf8c3a92aeb33d1e89306288052793df150
cache payload SHA  e713960bbcd26a943f907fc52a4f837b4144d15345c2d9f78a8d102b0728a01d
repeat run JSON    /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_native_cod_repeat_20260904.json
repeat run SHA     f1509870b71d6ed514017774db22858100acfc3f5ee9561a064295dc0f26179f
```

## Failure-driven changes

- 全員同じcatalogでは先頭候補へ寄ったため、既存runtime同様の人格別`role_preferences`を復元。
- 全人格の順番再投票をやめ、初期多数派を保持して異論側だけBase再計算。reconciliation callを4から2へ削減。
- 0.6Bはreasonとledgerを同じcallで渡すと追加キー・入力復唱・未入力事実を生成したため、再投票3キーとchange reason 1キーを分離。
- 変更理由は1.7B Base、本文だけLoRA。0.6Bと1.7Bを同時常駐させず、0.6B解放時にMLX activeが約335 MBから約3 KBへ戻ることを確認。
- Swiftの自動snake caseが`dataIDs`を`data_i_ds`にしたため、`CodingKeys`で`data_ids`へ固定。
- 本文cache不足時は開始thermalが`nominal`でなければモデルをロードせず中止する。全cache検証済みなら`fair`を許可する。

## Acceptance validator

```sh
python3.11 tools/validate_iphone_native_cod.py \
  /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_native_cod_20260904.json \
  --repeat /Volumes/data4/cod_model_weight/evaluations/claim-body-v3/iphone13_a15_native_cod_repeat_20260904.json
```

validatorは、synthetic表示、role preference、D番号、raw schema、LoRA非介入、選択的speaker、投票再集計、異議→変更→賛同順、change reason、本文cacheのraw/claim/digest、fallback 0、Adapter lifecycle、35秒以下、512 MiB以上のheadroom、thermal gateを検証する。

## Remaining boundary

このfixture PASSだけでExtremeWeatherへ統合しない。次に実データから小さなledgerを構築し、事実cutoff、source URL、D番号の意味監査、キャンセル、3D画面とのmemory/thermal共存を確認する。Weightの昇格範囲は引き続きClaim Body rendererだけで、0.6B構造モデルをWeightとして昇格したことにはしない。

## Historical Typhoon 18 replay readiness

次段階用に、既存の台風18号data packetから2026-08-25 15:50 JST cutoffの小型replay ledgerを作成した。現在予測ではなく歴史的再生であり、各D番号にsource ID、URL、観測時刻、有効期限を持たせた。親ledgerとpacketのSHAも固定している。

- ledger: [`../data/typhoon18_20260825/native_cod_replay_ledger.json`](../data/typhoon18_20260825/native_cod_replay_ledger.json)
- ledger SHA-256: `7f5f0f5032158ae448f97976bfd3ec1f336507a79a5634f554f23cce243825eb`
- source 5 / data 6 / claim 6 / persona 4
- 0.6B Mac blind preflight: direct 3/4、根拠ID再計算1回、final 4/4
- Claim Body v3 Mac preflight: contract valid 6/6、safe politeness sanitize 3/6、fallback 0
- readiness raw: `/Volumes/data4/cod_model_weight/evaluations/claim-body-v3/typhoon18_native_replay_readiness_20260904.json`
- readiness SHA-256: `8fa2662a6a602202b414b7a26700380574da8b38db64bc6205af7eea320bcf2a`

`tools/validate_native_cod_replay_ledger.py`がprovenance、時刻範囲、source、claim、contradiction、role preference、Base選択、本文意味一致を検証済み。物理iPhone runはまだ完了していない。
