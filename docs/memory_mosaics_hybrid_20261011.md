# Memory Mosaicsと再帰型ハイブリッドの研究実装

2026年10月11日、MP CoDの研究用toolsへ、学習可能な連想記憶層とMamba／RWKV7のハイブリッド比較を追加した。6種類の小規模CPUブロックで、因果性、別バッチへの情報非干渉、有限な勾配と2回の更新を確認した。これは言語モデルでも通常のCoD経路への採用でもなく、研究用の動作検証である。進行中のQwen3.5学習、Base、既定の推論経路は変更していない。

## 論文の結果と実装範囲

指定された[Memory Mosaics原論文](https://arxiv.org/abs/2405.06394)と、大規模版の[Memory Mosaics at scale](https://arxiv.org/abs/2507.03285)は別論文である。「1兆tokensのモデルが8兆tokensのTransformerを上回る」は、主に新知識の利用と文脈内学習の評価に関する結果で、全性能が常に上回ることや、推論コストが8分の1になることを意味しない。大規模版の32k長文質問応答の比較は53.4対46.9だが、4kでは58.9対59.2であり、意味のあるラベルの文脈内学習では8兆tokens版が追いつく結果もある。[比較条件](https://arxiv.org/html/2507.03285v3#S6)

連想記憶は、過去のキーと値を保持し、現在のキーに近い値を補間するニューラル層である。今回の実装は、対称なキー検索、入力依存の再帰型キー抽出、次の位置を使う値抽出、記憶件数に応じた帯域、独立した短期・長期記憶、SwiGLUによる恒久記憶を試す。次の位置を使う値は、その位置が観測済みになった過去の組だけから取り出すため、未来を出力へ漏らさない。[アーキテクチャ](https://arxiv.org/html/2507.03285v3#S3)

全文の言語モデル再現ではない。語彙の入出力、深い層の積み重ね、論文と同じ初期化や確率的な長期窓、事前学習は含まない。指数ゲートの上限、正規化の微分安定化、帯域指数の再パラメータ化等も、小規模検証の数値安全性を優先している。単なるJSON検索やRAGをMemory Mosaicsと呼び替えたものではないが、原論文の大規模な性能を再現したとも扱わない。

長期のキーと値を保持するこの方式は、記憶を固定サイズに圧縮するMamba／RWKVとは目的が異なる。今回は全文の記憶計算を行うため計算は二次的で、CPU・FP32・幅64以下・系列64以下・バッチ4以下の実験に制限する。長文脈の軽量化や完全な記憶を保証しない。[原論文の計算量](https://arxiv.org/html/2405.06394v3#S2)

## ハイブリッドの比較

Mambaは[選択的な状態空間モデル](https://arxiv.org/abs/2312.00752)、RWKVは[再帰型言語モデル](https://arxiv.org/abs/2305.13048)である。TransformerとMambaを組み合わせた大規模モデルには[Jamba](https://arxiv.org/abs/2403.19887)という先行例がある。今回のMambaとRWKV7は、インストール済みMLX-LMの実モジュールを再利用する。独自の平均処理をそれらのモデルと呼び替えていない。

| 構成 | 再帰型部分 | 検索部分 | 微小パラメータ数 |
| --- | --- | --- | ---: |
| attention | なし | causal Attention | 2,576 |
| mosaics | 記憶キーの再帰抽出 | 短期・長期の連想記憶 | 3,236 |
| mamba_attention | MLX-LM Mamba | causal Attention | 4,848 |
| mamba_mosaics | MLX-LM Mambaと記憶キー | 連想記憶 | 5,508 |
| rwkv7_attention | MLX-LM RWKV7 | causal Attention | 5,328 |
| rwkv7_mosaics | MLX-LM RWKV7と記憶キー | 連想記憶 | 5,988 |

全構成に恒久的な学習パラメータを持つ非線形層がある。この比較はパラメータ数を揃えていないので、損失や時間を順位付けして優劣を主張しない。Pythonモジュールの初回import時間も異なる。

現在のQwen3.5-4Bもすでにハイブリッドである。ローカルの実設定はGatedDeltaNet系linear attentionが24層、full attentionが8層だった。Mamba／RWKVそのものではない。[Qwen公式モデル](https://huggingface.co/Qwen/Qwen3.5-4B)

## 実際に確認したこと

既存MLX環境の7テストで、未来側の入力を書き換えても前半出力が変わらないこと、前半だけを計算した結果との一致、別バッチの変更が他バッチへ波及しないことを確認した。空の記憶、ゼロ入力の連想記憶勾配、サイズ・型・非有限値の拒否、失敗記録に昇格権限を付けないことも検査している。これはキャッシュを持つ実対話システム全体の隔離証明ではない。

検証CLIは、固定したランダムな隠れベクトルを前の位置へ写す代用課題を、各構成で2回更新した。全6構成で損失・勾配・更新後パラメータが有限で、実際にパラメータが変化した。日本語の生成、未学習課題への適応、議論の事実保持、人格分離を測った実験ではない。通常Pythonでの182テストは成功し、MLX専用6件は依存環境がないためスキップした。

最初の微小学習はRWKV7の初期キー倍率がゼロのため非有限勾配で停止した。同じ1入力の診断では、倍率を1で新規初期化すると非有限勾配8配列が0配列になり、forwardの損失は同じだった。研究用の新規初期化だけを変更し、ライブラリ本体と事前学習済みWeightは変更していない。全入力でのRWKV勾配安定性は未検証である。元の失敗journal・ソースを残し、NaNの置換では合格にしない。

## 使い方

MLXを含む既存の研究環境で次を実行する。出力とjob journalが存在する場合は再実行せず、新しい出力名を使う。

```sh
cd /Users/osaka/src/cod_model
/Volumes/data4/cod_model_weight/venv/bin/python -m unittest -v test_memory_mosaics_hybrid
/Volumes/data4/cod_model_weight/venv/bin/python tools/benchmark_memory_mosaics_hybrid.py \
  --out /Volumes/data4/cod_model_weight/evaluations/memory-mosaics-hybrid_20261011/new_probe.json
```

大きなモデルのダウンロード、GPU実験、既定モデルの置換、Weight公開は行わない。比較結果に成功と表示されても、`promotion_allowed`と`default_replacement_allowed`はfalseである。数値失敗時は、それまでの結果を保存して例外で終了する。内蔵・外付けの空き20GiBを守る既存のjob監視を使う。

実験記録は`/Volumes/data4/cod_model_weight/evaluations/memory-mosaics-hybrid_20261011`にある。最初の`optimizer_smoke.json.job.json`は失敗記録、`rwkv_initialization_diagnostic.json`は初期化の対照、`optimizer_smoke_v3.json`は最終の6構成比較である。ソースSHAとネイティブ実装SHAを記録し、元の比較を書き換えない。

## MP CoDへの採用条件

当面は現行Qwen3.5の学習・評価を優先する。追加メモリ層を試す場合は、別のWeightと計画で、小さな残差による追加・蒸留・共同学習を比較する。既存LoRAを適用するだけでアーキテクチャが変わるものではない。Memory Mosaicsのheadと、MP CoDの人格を同一視しない。

通常経路へ入れるには、実際の隠れ次元に対応する学習とWeight、Tokenizerと出力層、要求ごと・人格ごとの状態管理、増分cache、元モデルへ戻す対照が必要となる。未学習の日本語討論、意味と根拠の保持、既存合格の後退確認、実速度とメモリを比較し、既存の採用基準を満たすまで既定値を置換しない。iPhoneとSimulatorでの実行やGGUFへの変換も未検証である。

動画の字幕テキストはローカル抽出で確認した。[指定動画](https://www.youtube.com/watch?v=vczoV2L7Ib4)は背景説明として扱い、性能・数式の判断は一次論文を優先する。指定OpenReview URLは閲覧確認とAPI403で本文を取得できず、査読内容を確認済みとはしない。原論文と大規模版のarXiv本文、公式コードは別途確認した。
