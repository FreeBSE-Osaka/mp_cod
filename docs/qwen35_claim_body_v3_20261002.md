# Qwen3.5-4B Claim Body v3 時制と話者別生成の検証

## 結果

v2のMLP Adapterから、計画・進行中・完了・効果未確認を区別する32件で追加学習した。
64ステップ版は新しい16問で直接合格が9/16から14/16へ改善した。
以前の条件保持16問は11/16から13/16、さらに前の12問は9/12から11/12、既存15問は15/15を維持した。
Adapterサイズは754,632 bytes（約0.755MB）のまま。

ただし「試している」を「試み、検証済みです」に変える1件があり、**研究用HOLD**。
実走のclaim共有では以前の実行中化を避けられたが、話者別生成では4発言を時制検査が拒否した。
学習の進展と実走の未解決点をともに記録し、既定Weightの置換やWeight公開はしていない。

## 学習と凍結

新規の32件は検索方式、評価、授業、倉庫の架空事例で、各話者8人へ展開した256例。
v2のtrainからseed固定で抽出した復習256例と混ぜ、trainは512例。
新規のdevは8件・2topic、最終検証は16件・4topicで8話者を含む。
新規testと以前の条件保持testはtrainとのclaim重複なし。
以前の実走や最終検証の答えをそのままtrainへ戻していない。

| 設定 | 値 |
|---|---|
| Base | Qwen3.5-4B MLX 4bit |
| 初期Adapter | v2 mlp_step48 |
| 学習対象 | 最後4層のmlp.down_proj |
| rank / scale | 4 / 8 |
| learning rate | 2e-5 |
| batch / iteration | 1 / 64 |
| gradient checkpoint | 有効 |
| 最大入力token | train 259 / valid 265 / test 266 |
| MLX報告peak memory | 6.009GB |
| validation loss | 開始0.120 / step64 0.171 |

全64ステップを正常完走した。lossは8 batchの値で悪化したが、同じdev8問の生成で選定した。
前回版は直接5/8・語尾補正込み6/8、step16は4/8、step32は6/8、step48は5/8、step64は7/8。
devの時制・意味変化を確認してから、step64を最終生成前に`selection.json`へ固定した。
devには「試んでいます」という不自然な活用が残った。

## 最終生成と非回帰

| 検証集合 | v2の直接合格 | v3の直接合格 | v3の語尾補正込み |
|---|---:|---:|---:|
| 新規の時制16問 | 9/16 | 14/16 | 14/16 |
| 以前の条件保持16問 | 11/16 | 13/16 | 13/16 |
| v1時代の12問 | 9/12 | 11/12 | 11/12 |
| email EV bikeの既存15問 | 15/15 | 15/15 | 15/15 |

strict JSONは全ての集合で全件合格。同じ本文1問のAdapter着脱前後で、Baseのraw出力が完全一致した。
語句アンカーは凍結したまま採点し、後から数字を上げる変更はしていない。
例えば「負担が減ったかどうかは未確認」は意味を保つが、指定した「負担が減ったかは未確認」に一致せず不合格。
一方、次の出力は実質的な意味変化なので採用しない。

```text
入力: 散策経路を短くして負担の軽減を試している
出力: 散策経路を短くして、負担の軽減を試み、検証済みです。
```

「検証済み」は入力にない。実行中の行為から検証完了を推論しており、検査で拒否された。
合成された小規模データでの結果であり、任意の分野や表現の完全な正確さを証明するものではない。

## 8人討論とcacheの比較

同じ架空のSwift/Rust選定を、claim共有と話者別本文生成で実走した。
全8人が両案を選択でき、対案交換の15 event後に8人のすり合わせ投票がSwiftへ揃った。
初期判断8人分と投票8人分のrawは両方式で完全一致した。

| 実測 | claim共有 | 話者別 |
|---|---:|---:|
| 実行時間 | 208.175秒 | 273.036秒 |
| model calls | 20 | 33 |
| 本文生成calls | 2 | 15 |
| 本文cache hits | 21 | 8 |
| 公開23発言のWeight本文 | 23/23 | 19/23 |
| 本文の代替表示 | 0 | 4 |
| 失格claim | 1 | 1 |
| hard gate | fail | fail |

claim共有ではRust案が「将来の移植に備えます」と生成され、以前の「備えています」という実行中化は出なかった。
話者別では一部の話者に同じ実行中化が出て、凍結claim由来の代替表示へ戻った。
Swift案の本文は話者別でもほぼ同じであり、15回生成したこと自体を文体の多様化とは扱わない。
測定中の他の処理による速度変動があるため、全差分をcache方式だけの原因と断定しない。

Base側の未許可D05引用は残り、議論全体のhard gateは両方式ともfail。
本文が全てモデル由来のclaim共有も、議論全体の完全な検証成功とは言えない。
同様にBaseの理由文にある未確認の「安全性確保」は、判断rawの一致によって真実になるわけではない。

## 話者別生成オプション

`--body-cache-scope speaker`を追加した。本文のcache keyを話者IDとclaim labelにし、
同じ話者の再発言では再利用する。既定の`claim`は従来と同じ本文共有。
`speaker`には`--body-adapter`が必要で、無効な組合せはモデル読み込み前に拒否する。
一つのBaseとAdapterで実行し、8つのモデル本体をロードする機能ではない。

```sh
<mlx-python> cod_model.py event-debate --domain general --backend mlx \
  --model-path /path/to/Qwen3.5-4B-4bit \
  --ledger data/general_language_choice/claim_ledger.json \
  --body-adapter /path/to/evaluated-adapter --body-cache-scope speaker \
  --max-turns 24 --reconcile-rounds 2
```

現在は実験用。呼び出し数が増え、今回のWeightでは独自の文体と時制保持が十分でなかったため、既定に採用しない。

## 保存先と次の課題

```text
dataset: /Volumes/data4/cod_model_weight/datasets/claim-body-qwen35-4b-v3/mlx_shared
adapter: /Volumes/data4/cod_model_weight/adapters/claim-body-qwen35-4b-v3/step64
audit: /Volumes/data4/cod_model_weight/evaluations/claim-body-qwen35-4b-v3_20261002
```

[教材](../data/general_body_qwen35_v3/curated.json)と[設定](../configs/claim-body-qwen35-4b-v3.yaml)を保存した。
`evaluation_plan.json`、`train.log`、`selection.json`、各dev/fresh16/regression JSONと
`language_claim/`、`language_speaker/`のrun JSONで、元の生成・採否・cache・投票を確認できる。
再現時は未使用の出力先で、v2の`mlp_step48`を`--resume-adapter-file`に指定する。

時制だけでなく、未確認の効果を検証済みへ変えないことと、話者による不安定さが次の課題。
今回の最終検証や実走の返答を正解として次の学習へ混ぜず、別の教材と新しい検証を用意する。
