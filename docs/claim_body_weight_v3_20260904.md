# Claim Body Weight v3 / renderer contract v2 — 2026-09-04

## Result

Qwen3-1.7B-4bitのclaim本文専用LoRAを、現行の条件付きWeightとして昇格した。構造判断はBaseのまま、Weightは凍結済みclaimを丁寧な自然文へする処理だけを担当する。

- 外部3 topic、15ケース: contract valid `15/15`
- strict `bodies` schema: `15/15`
- 丁寧な完全文: `15/15`
- targetと完全一致する丁寧語化: `14/15`
- 競合claim選択: `0`
- email、EV、bikeの1 round実走: 全てWeight `6/6`、fallback `0`、hard gate通過
- Adapter Weight: 2,496,303 bytes
- Base置換、構造判断、自動HF公開は許可しない

## Why v1 was superseded

v1の再監査で、次の意味変化を当時の類似度validatorが通していた。

```text
claim: 自動triageを全mailへ展開する
bad:   自動triageを全mailへ展開した
bad:   削減が実現された
bad:   展開することを検証済み
bad:   削減を優先。
```

また、旧datasetのreconciliation targetには`『claim』を採ります`、根拠文だけ、`推定されます`が混ざり、本文専用契約と矛盾していた。v1の条件付き昇格を取り消し、次を追加した。

- proposalを`実現 / 完了 / 達成 / 検証済み`へ変える文を拒否
- `を採ります / を選びます`等のmove混入を拒否
- `です / ます`系の丁寧な完全文を必須化
- strict schema例と「IDをJSONキーにしない」をsystemへ明記

## Intermediate v2 stop

最初のclean datasetは6 topicを`train 64 / valid 16 / test 16`へ分け、[`configs/claim-body-v2.yaml`](../configs/claim-body-v2.yaml)でBaseから64 step学習した。test lossは`0.177`、さらに32 step継続すると`0.093`まで下がったが、現行contractでの直接validはstep64 `0/7`、step96 `1/7`だった。完了事実化、根拠置換、plain formが残ったため両方とも非昇格にした。

## Dataset and training

17 train topicの各claimを全参加人格へ割り当て、targetはclaim末尾の丁寧語化だけにした。topic単位でvalid/testを分離し、最終評価のemail、EV、bikeはdataset全体から除外した。

| Split | Topic | 件数 |
|---|---:|---:|
| Train | 17 | 585 |
| Valid | 1 | 27 |
| Test | 1 | 27 |
| External holdout | 3 | 15 |

学習設定は[`configs/claim-body-v3.yaml`](../configs/claim-body-v3.yaml)。Baseからlast 4 layers、rank 4、learning rate `2e-5`で128 step学習した。

| 指標 | 値 |
|---|---:|
| 初期valid loss | 1.739 |
| step 32 | 1.082 |
| step 64 | 0.236 |
| step 96 | 0.073 |
| step 128 | 0.039 |
| 最終train loss | 0.019 |
| test loss / perplexity | 0.062 / 1.064 |
| peak memory | 2.087 GB |

step64は外部15ケースで`8/15`だったため採用せず、step128だけを候補にした。

## Renderer contract v2

本文rendererにevidenceを渡すと、claimではなく根拠文を要約する失敗が発生した。evidenceはBaseが既に検証し、D番号・statement・ログへ保持しているため、renderer入力からだけ除外した。

v3 Weight自体はevidence付きinputで学習したが、外部評価でclaim-only入力への転移を確認した。後述のv4で学習inputも完全一致させたところvalidは同値、exact politeは悪化したため、実測が良いv3を維持した。

```json
{
  "items": [
    {
      "id": "B01",
      "speaker": "批判的設計者",
      "claim": "誤振分けが集中する契約添付mailでtriage pilotを先に行う"
    }
  ]
}
```

- 1 call 1 itemなのでtransport IDは常に`B01`
- 元のevent / reconciliation IDは`renderer_batches.record_id`へ保存
- Body Weightだけtemperature `0.0`
- 数字はclaimに含まれるものだけ許可
- moveはWeightへ生成させず、検証後にコードで合成
- 不合格時は検証済みstatementへfallback

## External transfer

claim-only契約でv1とv3を同一15ケース比較した。

| Candidate | Contract valid | Exact polite | Strict schema | Competing claim |
|---|---:|---:|---:|---:|
| v1 step64 | 0/15 | 0/15 | 15/15 | 0 |
| v3 step128 | 15/15 | 14/15 | 15/15 | 0 |

v1もclaimとの語彙一致だけなら15/15だったが、全件がplain formで、1件は名詞断片だった。v2契約では丁寧な完全文を要求するため0/15となる。v3の非完全一致1件は`利用不能が集中する寒冷depot`を`寒冷depot`へ短縮したが、方向・対象・時制を保った自然文としてvalidatorを通過した。

## Live round results

| Holdout | Public utterances | Weight | Repair | Fallback | Hard gate | Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Email triage | 6 | 6 | 0 | 0 | pass | 58.0秒 |
| EV smart charging | 6 | 6 | 0 | 0 | pass | 52.5秒 |
| Bike rebalancing | 6 | 6 | 0 | 0 | pass | 46.6秒 |

EVの公開発言は次のとおり。

```text
批判的設計者:
寒冷depotでsmart charging pilotを先に行います。

仮説構築者:
その結論には異議があります。代案として、電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開します。

実証監査者:
私も賛成です。特に、電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開します。

実行設計者:
その案には賛成です。加えて、電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開します。

仮説構築者:
結論は変わりません。電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開します。

批判的設計者:
考え直しました。電力費18%とCO2 11%の削減を優先してsmart chargingを全車両へ展開します。
```

4人格の投票はBaseが独立生成し、Weightは確定後の本文しか見ていない。

## Validated claim cache

同じclaimがeventとreconciliationで繰り返されても、本文は同じでmoveだけが異なる。valid済み本文をclaim label単位でrun内cacheし、ラウンドではコード側のagree / maintain / revise prefixだけを付け直すようにした。不合格本文はcacheしない。

同じEV 2 event / 1 roundを変更前後で比較した。

| Metric | Before | Cached | Change |
|---|---:|---:|---:|
| Body model calls | 6 | 2 | -66.7% |
| Total model calls | 14 | 10 | -28.6% |
| Elapsed | 52.5秒 | 41.9秒 | -20.2% |
| Weight utterances | 6/6 | 6/6 | unchanged |
| Hard gate | pass | pass | unchanged |

event発言、4人格の投票、表示文、変更理由、最終summaryは旧runと完全一致した。`renderer_batches.record_ids`へ全再利用先、各発言の`renderer_cached`、metricsへ`body_renderer_model_calls / body_renderer_cache_hits`を保存する。

reconciliationの初回statementで選択は有効だがD番号だけが不正な場合、独立主張ですでに使っている`sanitize_model_statement`を再利用する。sanitized文が選択claimへ競合案より強く一致し、競合claimを選んでいない時だけmodel repairを省く。見解変更理由が不正な場合は従来どおりrepairする。

Email 2 event / 1 roundでは、cache適用後のbaselineからさらに次のように短縮した。

| Metric | Cache only | Cache + sanitizer |
|---|---:|---:|
| Reconciliation repair calls | 4 | 1 |
| Total model calls | 14 | 11 |
| Elapsed | 49.6秒 | 45.1秒 |
| Weight utterances | 6/6 | 6/6 |
| Hard gate | pass | pass |

公開発言、投票、D番号、変更理由、summaryは完全一致した。metricsへ`reconciliation_model_repairs / reconciliation_statement_sanitizations`を保存する。

通常規模のEV 8 event / 最大2 roundでは、8件中1件だけWeightが凍結claimそのものをplain formで返した。bodyがlabelと完全一致する場合に限り、学習targetと同じ文末変換（`する→します`等）を適用する`model_body_v2_sanitized`を追加した。言い換え、部分一致、根拠文には適用しない。

| Metric | Before | Exact-claim sanitize |
|---|---:|---:|
| Public utterances | 12 | 12 |
| Weight utterances | 11 | 12 |
| Politeness sanitizations | 0 | 1 |
| Fallbacks | 1 | 0 |
| Hard gate | fail | pass |
| Total model calls | 16 | 16 |
| Elapsed | 49.3秒 | 54.2秒 |

表示文、D番号、投票、変更理由、summaryは完全一致した。時間差は実行揺らぎとして扱い、速度改善とは判定しない。

## v4 stop result

runtimeと完全一致するclaim-only datasetでv3から[`configs/claim-body-v4.yaml`](../configs/claim-body-v4.yaml)を使って32 step追加学習した。v4 step160は外部valid `15/15`を維持したが、exact politeが`14/15`から`13/15`へ下がったため非昇格とした。loss低下だけでは親を置換しない。

## CLI

```sh
<mlx-python> cod_model.py event-debate \
  --ledger <claim-ledger.json> \
  --domain general \
  --backend mlx \
  --model-path <Qwen3-1.7B-4bit> \
  --body-adapter /Volumes/data4/cod_model_weight/adapters/claim-body-v3/shared_step128 \
  --max-turns 12 \
  --reconcile-rounds 2
```

## Artifact hashes

```text
v3 dataset manifest  cdf48fbf221fa6f5cfd380a71ce99e2d89bd886a0f131ba47c685f3b66850a89
v3 training config   5a00d99bfd9fac51f530c17630f01bf1e874c107f50e694e21cf9a9a3d6521e9
v3 adapter config    c5263707f55c583bfa98e15d169d9199d30f57ebf343ba6b1e9784d7b4fbcea6
v3 adapter weights   4ce21e64af220f0ee309599e189fd136e10c4c5cd11440c3d60fd306749a9a92
v2 strict rescore    13ce7f36ab89ed2e7cc15c28efc6fbd9f18c5081868fa36a7771c0f16928f475
external 15 cases    8491a38789a22444eeb0942b9f5969a21fb3cf3ae5411757dc981173345e36cf
email runtime        9178d81a2be5ba218b8694fc252e39b281c7e952e7d0487e5b4957baf38dac03
EV runtime           fb3b2cc3b94226ebb4bb7b19c63d26b90722ed2c94fa3d58a0ec64a0b5b5a89c
Bike runtime         3424c66152ab6a225a7ca8ea304af7ab4a516a0362cd337525b5ff3bc9a90f1a
EV cached runtime    a56a7d06d2b2edb054f9c536d27adc11535caf5808747e80de3a66631641f87e
Email cache baseline 2295484814e24c80985cef05d019310ba32bb9ae9de34bf65defa16364f909a6
Email optimized      4d736dab81b03635c950b07311a49195cb094e865ebdc4e5f466e2d5f29dfdb8
EV full8 before      c7c1dc97e6f71eff771e20b18daf436cbc43a5b014094c81e1d96cc3aab6ecea
EV full8 sanitized   1ccd5472dd5ede2197deb383ea0e50ec5aa895fc076d0cf538b66d89a109a9e4
v4 dataset manifest  bc978b4d7ada314395c050cdba4121727465fc2841ff2966c109a2365a0a1376
v4 training config   5e8f1ff7e12d558c1b7d22df2a0d7fb5808ab54be0d38c00bd2c5447136a700f
v4 adapter weights   749b80dc2666061246f55d1b5a59012c6291da4646127bca1f750a16a354b9e9
v4 comparison        44a5bb52fa52b0495af0735855c6a26aed65bd1426e5a49854df9c66fe8f95b0
```

rawは`/Volumes/data4/cod_model_weight/evaluations/claim-body-v3/`、Weightは`/Volumes/data4/cod_model_weight/adapters/claim-body-v3/shared_step128/`に保存した。Gitにはコード、設定、評価境界だけを置く。

## Weather and iOS portable import

学習から除外したweather domainの台風18号12 event / 1 roundでも、公開16発言すべてWeight由来、fallback 0、hard gate passとなった。run JSONへledger snapshotと人格metadataを埋め込み、ExtremeWeatherの純Swift importerで再検証・Simulator表示まで完走した。詳細と画面証跡は[ExtremeWeather portable CoD import](extremeweather_portable_import_20260904.md)を参照。

## Physical iPhone MLX smoke

物理iPhone 13 Pro / A15 / iOS 17.6.1で、`Qwen3-1.7B-4bit`と本Adapterを直接ロードし、weather claim 1件をstrict JSONの丁寧な本文へ変換した。通常memory limitでAdapter unloadとsession clearまでPASSし、warm runはTTFT 2.306秒、25.732 tok/s、total 4.558秒、thermal nominalだった。result JSONのSHA-256は`c3008004f01ddf941f9b6080e1f8df869ca5c35e8f432a67616a4bebcf9bd8ba`。詳細は[iPhone 13 Pro / A15 device smoke](iphone13_a15_claim_body_v3_smoke_20260904.md)を参照。

これは本文renderer 1 callの成立確認であり、完全な複数人格CoD、連続round、Peak memory、ExtremeWeather本体との共存は未検証である。
