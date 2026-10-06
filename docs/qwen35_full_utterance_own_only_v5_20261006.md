# Qwen3.5 全文発言の本人限定再描画学習

General全文発言v5は、既存の均等学習へ「本人の凍結主張と資料だけで話す」64入力を追加する研究用LoRAである。会話再生成の実行形式と教材のずれを減らし、他人格の文章や相手の案をコピーせず、確定した自分の立場を自然に述べることを狙う。学習完走、品質向上、Weight昇格はまだ確認していない。

実学習は親Weightのload後、16/320 step、898正解tokensまで進み、MLXピーク14.310GB、train loss0.681で最初の救済用Weightを保存した。初回の開発8 batch lossは0.926。これは学習・保存経路の確認で、開発48入力の生成品質や、固定比較候補の選定ではない。

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

表は事前に同じseedとインストール済みiteratorの順序を再現した値で、実学習の結果ではない。native callbackの観測を照合し、最終的に全320表現を消費したことを確認する。学習は他の実推論と同時起動せず、内蔵と外付けの双方に20 GiB以上の容量余裕を求める。OS全体のwired設定や他アプリの状態は変更しない。

## 品質の選定と停止条件

現在の共通検査で保存済みrawを再採点した比較参照は、Baseが直接26/48・事実保持18/48、親が直接32/48・事実保持25/48。これは新しいBase生成ではなく、同一requestの原評価との照合付き再採点である。

4候補を同じ開発48入力、220-token上限、temperature0、seed固定で実生成比較する。選定順は、Baseの事実保持合格を失った入力が少ない、親の合格を失った入力が少ない、事実保持合格が多い、直接合格が多い、早いstepの順に固定する。

未学習生成へ進む条件は、Baseと親の事実保持合格を失わず、事実保持25/48を上回り、対案・異議の直接検査が各4/6以上で、原文の事実と自然さにも問題がないこと。満たさなければHOLDとし、未学習72入力を候補選びへ使わない。

旧全文18例・source16例、保存採点の非回帰、Adapter除去、同条件の8人実討論、真の見解変更と独立した同意理由の確認も必要である。lossだけの改善、検査の接続語修正、2発言だけの再生成成功は、Generalの完成やWeight昇格の証拠にはしない。

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
planned adapter: /Volumes/data4/cod_model_weight/adapters/general-utterance-qwen35-4b-v5/train320_own_only
```

`plan.json`のSHA256は`fe5b6533ec0ed94589800c2e362fbe0bbedfb76df83f9becdfd9a85ae84e4fe0`。ソース・モデル・親・教材・比較参照・選定条件を学習前に固定し、source snapshotを保存した。`token_and_coverage_audit.json`は実token列と予定順序の監査、`native_training_progress.json`は実学習callbackの記録、job journalは終了・失敗・容量停止の証跡とする。旧Weightと原評価は上書きしない。
