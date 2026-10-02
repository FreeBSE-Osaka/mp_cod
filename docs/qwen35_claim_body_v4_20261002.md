# Qwen3.5-4B Claim Body v4 行為・効果・話者の検証

## 結果

64ステップの追加学習を完走し、devの同点ルールでstep16を最終生成前に選定した。
Adapterは754,632 bytes（約0.755MB）。今回も**研究用HOLD、既定置換なし、Weight公開なし**。
小さいのはAdapterだけで、推論には別途Baseモデルが必要である。

以前の「試している→検証済み」という失敗は、同じ入力で修正できた。
しかし新規16問は旧構成と同じ14/16、確認済み事実の追加8問は8/8から5/8、
以前の12問は共通の改訂guardで11/12から9/12へ下がった。
8人の話者別実走でも5発言が提案を進行中の事実へ変えた。改善例だけで昇格しない。

## 学習データと選定

32件の架空の教材を8話者へ展開した256例に、v3 trainからseed固定で抽出した復習256例を混ぜた。
「行為を試す」「効果は未確認」「行為は完了」「効果も検証済み」を区別し、
確認済みの事実を一律に否定しない教材も含めた。
新しいdevは2 topic・8件、最終検証は4 topic・16件で8話者を含む。
既存のdev/test出力を教材として取り込んでいない。

最初の草案には「管理札を試みる」のような不自然な目的語があった。
草案の学習も64ステップ完走したが、dev/final生成前の教材レビューで候補から除外し、
「管理札の導入を試みる」のような行為表現に修正した。
修正版は別dataset・別Adapterディレクトリで親v3から学習し直した。
草案の教材、Weight、logは保存した。途中停止したと扱わない。

| 設定 | 修正版の値 |
|---|---|
| Base | Qwen3.5-4B MLX 4bit |
| 初期Adapter | v3 step64 |
| 学習対象 | 最後4層のmlp.down_proj |
| rank / scale | 4 / 8 |
| learning rate | 2e-5 |
| batch / iteration | 1 / 64 |
| gradient checkpoint | 有効 |
| 最大入力token | train 325 / valid 335 / test 336 |
| MLX報告peak memory | 7.024GB |
| validation loss | 開始0.039 / step64 0.088 |

step16・32・48・64はdevで全て8/8。
時制・効果状態の変化がないことを確認し、同点なら早いstepという事前ルールで16を選んだ。
最終検証の結果を見て別stepへ差し替えていない。

## プロンプトだけの効果とWeightの効果

同一Base・入力・話者・temperature 0で3条件を比較した。
新systemは「試している≠効果検証済み」「speaker名≠実施済みの根拠」を明示する。
表の値は直接合格。語尾補正込みも同じ値だった。

| 集合 | v3＋旧system | v3＋新system | v4 step16＋新system |
|---|---:|---:|---:|
| 新dev8 | 8/8 | 6/8 | 8/8 |
| 新規16問 | 14/16 | 8/16 | 14/16 |
| 確認済み事実8問 | 8/8 | 7/8 | 5/8 |
| 以前の時制16問 | 14/16 | 13/16 | 14/16 |

確認済み事実8問はstep選定後、生成前に凍結した追加検証で、学習・step選定には使っていない。
以前の時制集合の「新systemのみ」は、選定後の原因切り分け用の診断である。
新systemだけでは「試んでいます」や話者名の混入が増えた。
学習は新規16問の崩れを戻したが、旧構成全体を上回ってはいない。

例えば以前の失敗入力では、モデル本文が次のように変わった。

```text
入力: 散策経路を短くして負担の軽減を試している
v3: 散策経路を短くして、負担の軽減を試み、検証済みです。
v4: 散策経路を短くして、負担の軽減を試しています。
```

一方、確認済み事実の追加検証には、次の余分な話者属性が出た。

```text
入力: 音量設定の変更による聴取の負担軽減は検証済み
v4: 音量設定の変更による聴取の負担軽減は、実証監査者として、検証済みです。
```

確認結果を未確認へ反転した例は追加8問にはなかった。
5/8の失格3件は、この話者属性1件と、読点による固定アンカー不一致2件である。
新規16問の失格2件も意味を保つ言い換えだった。
ただし、生成結果を見てアンカーを緩めたり、合格値を書き換えたりしていない。
合成された小規模の検証で、任意の文章の意味保持・自然さを証明するものではない。

## 既存集合と語尾guard

「仮の案内だけを出す、です。」を既存の語尾検査が通していた。
共通の`body_is_polite_sentence`で、語尾の「です」直前の読点・空白を取り除いてから
既存の常体動詞＋です検査へ渡すようにした。動詞の不正活用を読点で回避できなくした。
正しい「高いです」「使わないです」等の扱いは変えない。

原rawと元採点は保存し、修正後の採点は別の`*_guard_v4.json`へ出力した。
凍結した意味アンカーは変えず、親と候補へ同じguardを適用した。

| 集合 | v3＋旧system | v4＋新system |
|---|---:|---:|
| 条件保持16問 | 13/16 | 13/16 |
| v1時代の12問 | 11/12 | 9/12 |
| email / EV / bike既存15問 | 15/15 | 15/15 |

候補の旧12問は元採点10/12だったが、不正活用1件を追加拒否して9/12になった。
別の1件は「登録者に限る」という条件を落とし、文法も崩した。
検査の直接合格だけで自然な会話と認定しない。校正や意味検査には依然として限界がある。

同じ本文1問を、Adapter着脱前後のBaseで決定論的に生成し、rawが完全一致した。
これは1問の本文probeであり、全判断・投票の非回帰を証明するものではない。

## 8人討論の実走

架空のSwift/Rust選定を、全8人が両案を選べる条件で実走した。
判断temperature 0.1、本文temperature 0、seed 20260825、最大24 event、最大2 round、話者別cache。
15 eventと8人の投票で、1 round後にSwiftへ揃った。人数の同意を独立した証拠とは数えない。

| 実測 | 候補の話者別実走 |
|---|---:|
| 実行時間 | 287.106秒 |
| model calls | 32 |
| 本文生成calls / cache hits | 15 / 8 |
| 公開23発言のWeight本文 | 18/23 |
| 凍結claim由来の代替表示 | 5 |
| 失格claim | 1 |
| hard gate | fail |

仮説構築者のRust本文rawは、例えば次の文だった。

```text
最初からRustの共通コアとSwiftUIを使い、初期工数を増やし、将来の移植に備えています。
```

入力は提案の「備える」であり、進行中の「備えています」は採用せず、凍結claimから代替した。
代替文をWeight生成成功に数えていない。
同意・対案の接続部分はコードが合成し、本文だけがWeight生成である。
話者別に15回生成しても文体の差は小さく、自然な自由討論の完成とは扱わない。

Baseの未許可D05引用は1件拒否した。
理由rawにはD05を「移植不要」、未実測のD06を「安全性確保」と扱う箇所も残る。
D番号や選択コードの形式検査だけでは、理由文の全ての意味的裏付けを保証できない。

以前のv3実走との初期raw一致は2/8、投票raw一致は2/8、最終選択は8/8で同じだった。
違いはAdapterを初めて有効化する前の初期生成にもあり、LoRA汚染の根拠にはできない。
旧runは判断temperatureの記録がないため、この比較を厳密な条件一致のWeight非回帰とはしない。
今回、今後のrunには`execution.decision_temperature`を保存するようにした。

## 実験指定と再現

既定systemや既定Weightは変えていない。
実験には`--body-system-file`を明示し、同じsystemを学習・評価・runtimeで使う。
空のsystem、body Adapterなしの指定はモデル読み込み前に拒否する。
CLIの本文systemはstrip後のSHAもrunへ記録する。ファイル全体のSHAとは別である。

```sh
<mlx-python> tools/general_body_training.py build \
  --curated data/general_body_qwen35_v4/curated.json \
  --renderer-system-file configs/claim-body-qwen35-4b-v4-system.txt \
  --rehearsal /path/to/v3/mlx_shared --rehearsal-train-limit 256 \
  --out /path/to/new-v4-dataset

<mlx-python> -m mlx_lm lora --train \
  --model /path/to/Qwen3.5-4B-4bit \
  --resume-adapter-file /path/to/v3/step64/adapters.safetensors \
  --data /path/to/new-v4-dataset \
  --config configs/claim-body-qwen35-4b-v4.yaml \
  --adapter-path /path/to/new-v4-adapter

<mlx-python> tools/general_body_training.py evaluate \
  --model /path/to/Qwen3.5-4B-4bit --adapter /path/to/selected-v4-adapter \
  --renderer-system-file configs/claim-body-qwen35-4b-v4-system.txt \
  --curated data/general_body_qwen35_v4/curated.json --split test \
  --check-adapter-isolation --out /path/to/new-final.json

<mlx-python> cod_model.py event-debate --domain general --backend mlx \
  --model-path /path/to/Qwen3.5-4B-4bit \
  --ledger data/general_language_choice/claim_ledger.json \
  --body-adapter /path/to/selected-v4-adapter \
  --body-system-file configs/claim-body-qwen35-4b-v4-system.txt \
  --body-cache-scope speaker --temperature 0.1 --seed 20260825 \
  --max-turns 24 --reconcile-rounds 2 --portable-context --out /path/to/new-run
```

再現では64の最終ファイルではなく、選定した`0000016_adapters.safetensors`とconfigを
別ディレクトリへ保存して評価する。新しい候補の昇格判断には、新しい未学習testが必要である。

## 保存先と確認範囲

```text
dataset: /Volumes/data4/cod_model_weight/datasets/claim-body-qwen35-4b-v4/mlx_corrected
adapter: /Volumes/data4/cod_model_weight/adapters/claim-body-qwen35-4b-v4/corrected_step16
audit: /Volumes/data4/cod_model_weight/evaluations/claim-body-qwen35-4b-v4_20261002
Weight SHA256: 7f58610e3cb3aec9b512fd404974addb81b2dddbb0fe51c509d4fa75f8859a04
```

[教材](../data/general_body_qwen35_v4/curated.json)、[追加8問](../data/general_body_qwen35_v4/confirmed_holdout.json)、
[system](../configs/claim-body-qwen35-4b-v4-system.txt)、[学習設定](../configs/claim-body-qwen35-4b-v4.yaml)、
[非昇格記録とハッシュ](../promotions/qwen3.5-4b-claim-body-v4-step16.json)を保存した。
auditには草案修正の経緯、修正版の凍結記録、学習log、選定、各条件のraw、改訂guard採点、
話者別runと`runtime_reference_comparison.json`がある。旧artifactを上書きしていない。

共通コードの単体テスト57件を確認した。
以前のiPhone用保存JSONも現行validatorで通過したが、今回のWeightを実機へ配備・実行した確認ではない。
次の課題は、system長大化の影響をdevで切り分け、利用条件と自然な活用を落とさず、
話者別でも提案を実施済みへ変えないこと。自由な表現の強化は、事実保持の非回帰とセットで扱う。
