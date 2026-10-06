# Qwen3.5 全文発言の本人限定再描画学習

General全文発言v5は、既存の均等学習へ「本人の凍結主張と資料だけで話す」64入力を追加する研究用LoRAである。会話再生成の実行形式と教材のずれを減らし、他人格の文章や相手の案をコピーせず、確定した自分の立場を自然に述べることを狙う。320 stepの1巡学習と固定4候補の開発生成比較は正常終了した。事前規則で選んだstep160は固定事実保持検査20/48で親25/48を下回り、個別後退と対案の失敗も残るため研究HOLDとする。既定Weightの置換とWeight公開は行わない。

実学習は2026-10-06に終了コード0で完走し、320 step、18,197正解tokens、固定候補80・160・240・320の保存を照合した。容量監視を含む実行は9,109.166秒、MLXピーク16.478GB、最終train loss0.357、最終開発8 batch loss0.863だった。lossと保存経路の確認は、開発48入力の発言品質や選定の成功とは別である。

## 既存教材と評価を保つ

`tools/general_utterance_training.py build --own-only-rehearsal`を任意で指定する。既定は無効である。[v4の均等教材](qwen35_full_utterance_balanced_v4_20261006.md)256表現をすべて残し、8演者と8発話行為の64組へ本人限定の1表現を追加する。全320表現は各組5表現ずつで、10学習topicを維持する。

追加表現は主張、phase、move、資料、speaker、speech act、入力IDだけを持つ。target claim、previous claim、候補理由、他演者の文章、perspectiveは含めない。source-groundedの本人限定再描画systemと同じ規則を使い、元の人為的な正解文と資料を変更しない。goldに実際の入力で裏付けられない数字があれば拒否する。

開発48表現と未学習72表現はv4のファイルとバイト単位で一致し、追加入力はtrainだけに入る。未学習72入力のモデル出力は生成しない。教材はCodex作成の架空例で、監修済み実議論や、評価で成功したモデル出力を集めたものではない。

実TokenizerとMLX-LM ChatDatasetで全440表現を確認した。prompt prefixと正解spanが一致し、EOSを含む正解位置を保持する。最大tokenはtrain722、valid706、test720で上限768以内。学習対象は18,197正解tokensで、実入力の切り捨てはない。入力・split保持・64組の配分・追加の本人限定表現を含む142単体テストが通過した。

## 学習条件を固定する

BaseはQwen3.5-4B-4bit、親は研究HOLDのv4 step192。親のSHA256は`69b0afa4116ab13b523c5e8d7539ceb3b490cbf30f31bce1d6332570e5a988f6`である。親を採用済みWeightへ昇格させる意味ではない。

[設定](../configs/general-utterance-qwen35-4b-v5-own-only.yaml)は、最後4層の`mlp.down_proj`、rank4、scale8、learning rate1e-4、batch1、320 stepの1巡。思考なし、assistant-only loss、EOS保持、未使用tail paddingだけの削除、勾配checkpoint、プロセス限定cache上限0を維持する。16 stepごとの保存は救済用で、比較候補は80・160・240・320だけに固定する。

| step | 順序再現で観測する演者と発話行為の組 | 入力表現 | 予定正解tokens |
| --- | --- | --- | --- |
| 80 | 50/64 | 80/320 | 4546 |
| 160 | 60/64 | 160/320 | 9155 |
| 240 | 64/64 | 240/320 | 13628 |
| 320 | 64/64 | 320/320 | 18197 |

表は事前に同じseedとインストール済みiteratorの順序を再現した予定値である。native callbackの実観測も予定の正解token数と一致し、320 stepの1巡を完了した。学習は他の実推論と同時起動せず、内蔵と外付けの双方に20 GiB以上の容量余裕を求める。OS全体のwired設定や他アプリの状態は変更しない。

## 保存した候補Weightと実学習の照合

実callbackの80・160・240・320 stepで観測した正解token数は、上の予定値と全て一致した。終了後に各Weightと同じAdapter設定を別ディレクトリへ複製し、SHA256を照合した。各Adapterは754,632 bytesで、Qwen3.5-4BのBaseは別途必要である。

| step | Adapter WeightのSHA256 |
| --- | --- |
| 80 | `04f9a94381a6286369be3443fbee3b98a2bdab842017359d850edbae56ae7bf6` |
| 160 | `2360dfda61f58be4a2243af0dbfaec628d9322635341884a539b99c2ee1364b1` |
| 240 | `132eac66748080f7e93c0931c8359f8520ffe62b3eefaf799580d23b7da8ef07` |
| 320 | `56a20445e421dfcf43988546d73195107a68a988fe2c285559285a380c6b020d` |

共通の`adapter_config.json`のSHA256は`092429aa17850d4655c6eab7a9247792de20e92957b890a7ce81d3f35c278a63`。`training.json`と終了journalを保存し、元学習プロセスの終了後に比較生成を開始した。4候補の評価は直列で実行し、比較の途中でソース・教材・検査・選定条件を変更しない。

最終保存後の終了処理には数分を要した。終了前のmacOS `sample`は、メインスレッドでMLXの`compile_erase`と`CompileCache`・arrayの解放を記録し、OSのphysical footprintを62.9G、同ピークを88.5Gと報告した。これはMLXが報告したGPU allocation peakの16.478GBとは別の指標であり、同一視しない。実際の終了理由や単なるプロセス停止の推測に代えず、終了コード0まで待った。実入力長ごとのコンパイルキャッシュが速度やメモリへ与える影響は、次の独立した小規模比較で確認する調査事項である。

同じtrain320入力だけを実Tokenizerとnativeのbatch iteratorでCPU診断すると、実長の入力形状は158種類、nativeの32-token単位のpadding後は11種類だった。後者の未使用paddingは計5,380 tokensで、正解18,197 tokensの範囲は変わらない。`training_shape_diagnostic.json`はモデルをloadせず入力形状だけを測定したもので、コンパイルキャッシュの実entry数・メモリ・学習速度を比較した結果ではない。学習条件は変更していない。

## 品質の選定と停止条件

現在の共通検査で保存済みrawを再採点した比較参照は、Baseが直接26/48・事実保持18/48、親が直接32/48・事実保持25/48。これは新しいBase生成ではなく、同一requestの原評価との照合付き再採点である。

4候補を同じ開発48入力、220-token上限、temperature0、seed固定で実生成比較する。選定順は、Baseの事実保持合格を失った入力が少ない、親の合格を失った入力が少ない、事実保持合格が多い、直接合格が多い、早いstepの順に固定する。

未学習生成へ進む条件は、Baseと親の事実保持合格を失わず、事実保持25/48を上回り、対案・異議の直接検査が各4/6以上で、原文の事実と自然さにも問題がないこと。満たさなければHOLDとし、未学習72入力を候補選びへ使わない。

旧全文18例・source16例、保存採点の非回帰、Adapter除去、同条件の8人実討論、真の見解変更と独立した同意理由の確認も必要である。lossだけの改善、検査の接続語修正、2発言だけの再生成成功は、Generalの完成やWeight昇格の証拠にはしない。

## 固定4候補の開発生成結果

全4候補は同じ開発48入力を生成し、それぞれ終了コード0で完了した。元のrequestと原文、Weight SHA、ソースSHA、保存済みBase・親の同一requestを照合し、事前に固定した選定順でstep160を選んだ。直接合格が多いstep240より、親の事実保持合格を失う入力が少ないためである。

| 候補 | 完全JSON | 直接検査 | 固定事実保持 | Base合格の喪失 | 親合格の喪失 | 対案の直接検査 | 異議の直接検査 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| step80 | 47/48 | 24/48 | 14/48 | 10 | 12 | 0/6 | 3/6 |
| step160 | 46/48 | 29/48 | 20/48 | 5 | 8 | 0/6 | 4/6 |
| step240 | 47/48 | 32/48 | 20/48 | 5 | 10 | 0/6 | 3/6 |
| step320 | 45/48 | 32/48 | 19/48 | 6 | 10 | 0/6 | 4/6 |

step160はBaseの18/48を集計では上回るが、Baseが通っていた5入力を失った。丸札の比較結果17秒・5台という条件の欠落、改善案への否定、試行継続で未確認状態を落とすケースなどがある。親の25/48と個別合格の保持も満たさない。集計の増加だけを品質改善とは扱わない。

これらは凍結した共通検査による値で、全原文の意味と自然さの合格を示すものではない。未学習72入力のモデル出力、旧全文18例・source16例の候補再生成、Adapter除去、候補を使った8人実走へは進めていない。`selection.json`と4つの原評価を保存し、[非昇格記録](../promotions/qwen3.5-4b-full-utterance-v5-step160.json)へ未達理由を残した。理由再生成の完成例を避ける修正は、比較完了後の別工程として扱う。

## 保存先と再現

```sh
python3.11 tools/general_utterance_training.py build \
  --curated data/general_utterance_qwen35_v2/curated.json \
  --balanced-pairs --profiles source_grounded flexible_plain structured_plain \
  --own-only-rehearsal --out /path/to/new-dataset
```

```text
dataset: /Volumes/data4/cod_model_weight/datasets/general-utterance-qwen35-4b-v5/mlx_own_only_epoch
audit: /Volumes/data4/cod_model_weight/evaluations/general-utterance-qwen35-4b-v5_20261006
training adapter: /Volumes/data4/cod_model_weight/adapters/general-utterance-qwen35-4b-v5/train320_own_only
candidate adapters: /Volumes/data4/cod_model_weight/adapters/general-utterance-qwen35-4b-v5/step{80,160,240,320}_own_only
```

`plan.json`のSHA256は`fe5b6533ec0ed94589800c2e362fbe0bbedfb76df83f9becdfd9a85ae84e4fe0`。ソース・モデル・親・教材・比較参照・選定条件を学習前に固定し、source snapshotを保存した。`token_and_coverage_audit.json`は実token列と予定順序の監査、`native_training_progress.json`は実学習callbackの記録、job journalは終了・失敗・容量停止の証跡とする。終了時のstackは`after_final_save_sample.txt`、入力形状のCPU診断は`training_shape_diagnostic.json`へ分けた。4候補の原評価は`step{80,160,240,320}_dev48.json`、事前規則の選定は`selection.json`に保存した。旧Weightと原評価は上書きしない。
