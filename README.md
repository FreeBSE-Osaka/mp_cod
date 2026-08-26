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
- 内部の証拠文 `statement` と、表示専用の会話文 `utterance` を分離
- 異議・賛同は相手の原文ではなく構造化主張だけを見て会話調で応答
- 証拠のない主張は自動失格し、対立は最大ラウンド内で3/4に達した時点で終了
- 人格ごとに効用と損失を分け、反論は前提否定・反例・トレードオフの型を使用
- prompt/configだけを比較するbounded RSI shadow gateを同梱
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
  --max-turns 12 \
  --reconcile-rounds 2 \
  --max-tokens 600 \
  --prompt-profile orthogonal_fewshot
```

モデルには他人格の文章を渡さず、検証済みの主張コードと証拠IDだけを共有します。採決に使うのは検証済みコードだけです。

- `statement`: D番号付きの内部証拠文
- `utterance`: D番号や内部codeを読まない、UI・実況用の会話文

異議なら相手の見落とし、賛同なら追加観点、見解変更なら「確かに／見落としていた／見方を改める」を発言契約として検査します。相手の原文は見せず、対象claimのlabelだけを渡すため、文章コピーによる擬似合意を避けます。

不正なD番号は拒否し、選択済みDだけで `statement` を一度修復します。本文は使えてD表記だけが欠けた場合は `model_sanitized`、会話生成に失敗した場合だけ証拠文由来の表示へ戻します。初回raw、反応raw、修復rawはすべて保存します。

会話例:

```text
力学役: 現時点では、北東転向外れは低位に扱う見方です。
アンサンブル役: ただ、その見方では独立シナリオとして残す可能性を十分に扱えていません。
観測役: 私もその見方に賛同します。独立シナリオとして残すことが重要です。
力学役: 確かに、その点を見落としていました。見方を改めます。
```

`--prompt-profile` は `baseline`、`orthogonal`、`orthogonal_fewshot` から選べます。既定値は目的関数と1件の形式例を使う `orthogonal_fewshot` です。

### 人格別LoRA

同じベースモデルを常駐させたまま、人格ごとにLoRA層を除去・ロードできます。

```json
{
  "schema_version": 1,
  "adapters": {
    "dynamical_modeler": "/path/to/dynamical-adapter",
    "ensemble_probabilist": "/path/to/ensemble-adapter"
  }
}
```

```sh
<mlx-python> cod_model.py event-debate \
  --ledger <claim-ledger.json> \
  --domain weather \
  --model-path <base-model> \
  --adapter-map <adapter-map.json>
```

各Adapterは `adapter_config.json` と `adapters.safetensors` が必須で、実行ログに両方のSHA-256を保存します。KVキャッシュは人格間で共有しません。

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

## bounded RSI shadow

RSIは、異なる固定ledgerを使ったdevelopmentとholdoutの両方で候補runがParentを上回るか検査します。

```sh
python3.11 cod_model.py rsi-shadow \
  --parent-dev runs/parent-dev.json \
  --candidate-dev runs/candidate-dev.json \
  --parent-holdout runs/parent-holdout.json \
  --candidate-holdout runs/candidate-holdout.json \
  --round 1 --max-rounds 3 \
  --out runs/rsi-round1.json
```

評価対象はモデル主張率、証拠文・会話文成功率、reaction失敗率、fallback率、生出力保存率、会話同文率、発言イベントの多様性です。holdoutへ改善が移らない場合は停止します。結果が良くても `promotion_allowed=false`、`parent_replacement_allowed=false` であり、人間確認なしにWeight、コード、GitHub、Hugging Faceを変更しません。

## テスト

```sh
python3.11 -m py_compile cod_model.py test_cod_model.py
python3.11 -m unittest -v
```

## ライセンス

コードは [MIT License](LICENSE) です。外部モデル、学習済みWeight、データソースには、それぞれのライセンス・利用条件が適用されます。
