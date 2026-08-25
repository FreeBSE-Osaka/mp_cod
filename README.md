# Multiple Personality CoD (Chain of Discussion)

[![tests](https://github.com/FreeBSE-Osaka/mp_cod/actions/workflows/tests.yml/badge.svg)](https://github.com/FreeBSE-Osaka/mp_cod/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一つのローカルLLMを、学派・方法論・リスク選好が異なる複数の専門家人格として独立に呼び出し、反論と擦り合わせを経て結論を作るCoD実験です。

ここでいう「Personality」はソフトウェア上の専門家ロールであり、精神医学的な概念を指しません。また、これはニューラルネットワーク内部のMoEではなく、複数人格の推論を調停するオーケストレーターです。人格別LoRA Weightは任意で追加できます。

内部の逐語的な思考（Chain of Thought）は扱わず、外部検証できる主張・前提・反論・証拠IDだけを記録します。

## 特徴

- 各人格は、他者の文章を見ずに初期見解を独立生成
- 通常討論は、相互反証 → 司会 → 独立監査まで実行
- イベント討論は、順番制ではなく `異議あり！` → `賛同＋補足` → 新規主張の優先順で発言
- イベント討論では、固定台帳にある主張コードと `D01` 形式の証拠IDだけを許可
- 証拠のない主張は自動失格し、対立は複数ラウンドで擦り合わせ
- 全呼び出しをJSON保存し、人間が承認した発言だけをLoRA用JSONLへ変換

## 必要環境

- Python 3.11
- 通常討論: 起動済みの [Ollama](https://ollama.com/) とローカルモデル
- MLX討論・Weight評価: Apple Silicon と [MLX-LM](https://github.com/ml-explore/mlx-lm)

通常討論・データ生成・テストに追加Pythonパッケージはありません。

## クイックスタート（Ollama）

```sh
git clone https://github.com/FreeBSE-Osaka/mp_cod.git
cd mp_cod

ollama pull qwen3.5:4b
python3.11 cod_model.py list --domain software

python3.11 cod_model.py debate --domain software \
  "新しいローカル検索アプリをSwiftで作るべきか、RustとWeb UIを組み合わせるべきか"
```

既定モデルは `qwen3.5:4b`、既定APIは `http://127.0.0.1:11434/api/chat` です。別モデルや別URLは `--model`、`--api-url` で指定できます。

実行中は各人格の公開発言を実況し、完全な構造化ログを `runs/` に保存します。途中失敗時は `.partial.json` が残ります。

## 証拠ID付きイベント討論（MLX）

サンプル台帳には、データ本文、許可する主張コード、各主張を裏付ける `D` 番号、対立関係、人格ごとの関心領域が入っています。

```sh
python3.11 -m venv .venv
.venv/bin/pip install "mlx-lm[train]"

.venv/bin/python cod_model.py event-debate \
  --ledger data/typhoon18_20260825/claim_ledger.json \
  --domain weather \
  --model-path mlx-community/Qwen3-1.7B-4bit \
  --max-turns 10 \
  --reconcile-rounds 2
```

モデルには他人格の文章を渡さず、検証済みの主張コードと証拠IDだけを共有します。これにより、他者の回答をコピーした擬似的な合意を減らします。

台風データは2026年8月25日15時50分JST時点の再現用スナップショットで、現在の予報には使えません。

## 学習データの承認と出力

成功しただけのログは学習へ入りません。内容を確認し、採否と使用する発言を明示します。

```sh
python3.11 cod_model.py mark runs/<run>.json approved --calls 2,5,8 \
  --note "技術名、根拠、反論後の修正を確認済み"

python3.11 cod_model.py mark runs/<run>.json rejected \
  --note "未確認の固有名と誤った計算が混入"

python3.11 cod_model.py export --runs runs --out data/sft
```

`approved` 以外はexportされず、承認済みrunでも `--calls` にない発言は除外されます。出力先は `data/sft/<domain>/<persona>/` です。

## 軽量Weight実験

Pythonが正解を厳密生成する有限集合カリキュラムを作れます。

```sh
python3.11 cod_model.py curriculum --count 240 --out data/auditor_curriculum
```

実験済みの `Qwen3-1.7B-4bit + empirical_auditor LoRA step 8` は、決定論的ツール証拠がある場合だけ条件付き昇格しています。証拠なしの汎用推論Weightではありません。

```sh
<mlx-python> cod_model.py weight-audit \
  --model <mlx-model-or-hf-repo> \
  --adapter <adapter-directory> \
  --upper 100 --condition 2 --target 3 --hypothesis 1/3
```

Adapter WeightはこのGitリポジトリに含めていません。再現条件と評価値は [実験記録](docs/weight_experiment_20260825.md)、ハッシュと運用境界は [昇格記録](promotions/qwen3-1.7b-auditor-r1-step8.json) にあります。

## テスト

```sh
python3.11 -m py_compile cod_model.py test_cod_model.py
python3.11 -m unittest -v
```

## ライセンス

コードは [MIT License](LICENSE) です。外部モデル、学習済みWeight、データソースには、それぞれのライセンス・利用条件が適用されます。
